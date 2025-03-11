import os
import pdfplumber
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import docx
import logging
import layoutparser as lp
import easyocr
import numpy as np
from typing import Optional, Union, List

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize models at module level for reuse
try:
    LAYOUT_MODEL = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config')
    logger.info("Successfully initialized LayoutParser model")
except Exception as e:
    logger.error(f"Failed to initialize LayoutParser model: {str(e)}")
    LAYOUT_MODEL = None


def enhance_image(image: Image.Image) -> Image.Image:
    """
    Apply common image enhancement techniques to improve OCR results.
    
    Args:
        image (Image.Image): The input image to enhance
        
    Returns:
        Image.Image: The enhanced image
    """
    # Convert to grayscale
    image = image.convert('L')
    
    # Enhance contrast
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)
    
    # Apply thresholding
    threshold = 150
    image = image.point(lambda p: p > threshold and 255)
    
    # Apply filtering to remove noise
    image = image.filter(ImageFilter.MedianFilter())
    
    return image


def preprocess_image(image: Image.Image, min_size: int = 1000, resize_factor: float = 1.5) -> Image.Image:
    """
    Preprocesses an image for OCR by applying common enhancement techniques.
    
    Args:
        image (Image.Image): The input image to enhance
        min_size (int): Minimum dimension size for resizing
        resize_factor (float): Factor by which to resize the image if needed
        
    Returns:
        Image.Image: The preprocessed image
    """
    # Resize image if needed
    if min(image.width, image.height) < min_size:
        image = image.resize(
            (int(image.width * resize_factor), int(image.height * resize_factor)),
            Image.Resampling.LANCZOS
        )
    
    # Apply common enhancement techniques
    return enhance_image(image)


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from all pages of a PDF file.
    
    Args:
        file_path (str): Path to the PDF file.
        
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        return ""
    except (pdfplumber.PDFSyntaxError, pdfplumber.pdfminer.pdfdocument.PDFEncryptionError) as e:
        logger.error(f"PDF parsing error for {file_path}: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error extracting text from PDF {file_path}: {str(e)}")
        return ""


def advanced_extract_text_from_pdf(file_path: str, mode: str = "full", max_pages: int = 3, 
                                  keywords: Optional[List[str]] = None) -> str:
    """
    Extract text from a PDF using LayoutParser's pre-trained detection model to identify text blocks
    and applying pytesseract OCR on those regions.
    
    Args:
        file_path (str): Path to the PDF file.
        mode (str): Extraction mode - "full" processes all pages, "optimized" processes only first N pages
                    and filters out non-text elements.
        max_pages (int): Maximum number of pages to process when in "optimized" mode.
        keywords (Optional[List[str]]): List of keywords to filter text in optimized mode. Text segments
                                        that don't contain any of these keywords will be skipped.
        
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    try:
        if LAYOUT_MODEL is None:
            logger.error("LayoutParser model not initialized, falling back to standard extraction")
            return extract_text_from_pdf(file_path)
            
        with pdfplumber.open(file_path) as pdf:
            # Determine which pages to process based on mode
            if mode.lower() == "optimized":
                # Process only first N pages
                pages_to_process = pdf.pages[:max_pages]
                logger.info(f"Using optimized mode: processing first {len(pages_to_process)} of {len(pdf.pages)} pages")
            else:  # "full" mode
                # Process all pages
                pages_to_process = pdf.pages
                logger.info(f"Using full mode: processing all {len(pages_to_process)} pages")
            
            # Process the selected pages
            for i, page in enumerate(pages_to_process):
                try:
                    image = page.to_image()
                    layout = LAYOUT_MODEL.detect(image)
                    page_text = ""
                    
                    for block in layout:
                        try:
                            # Get block type/category (e.g., "text", "table", "figure", "header", "footer")
                            block_category = block.type if hasattr(block, "type") else str(block.label)
                            
                            # In optimized mode, filter out non-text elements
                            if mode.lower() == "optimized":
                                # Skip headers, footers, and images
                                if any(category in block_category.lower() for category in ["header", "footer", "figure", "image", "table"]):
                                    logger.debug(f"Page {i+1}: Skipping block of type '{block_category}'")
                                    continue
                                
                                # Only process text blocks
                                if "text" not in block_category.lower():
                                    logger.debug(f"Page {i+1}: Skipping non-text block of type '{block_category}'")
                                    continue
                            
                            # Extract text from the selected block
                            segment_image = image.crop(block.coordinates)
                            segment_text = pytesseract.image_to_string(
                                segment_image,
                                config='--psm 6 --oem 3'  # Page segmentation mode 6 (single block of text)
                            )
                            
                            # Add the segment text if it's not empty
                            if segment_text.strip():
                                page_text += segment_text + "\n"
                        except Exception as e:
                            logger.warning(f"Error processing block in page {i+1} of {file_path}: {str(e)}")
                            continue
                    
                    # In optimized mode, check if page contains any of the keywords
                    if mode.lower() == "optimized" and keywords and page_text:
                        # Convert to lowercase for case-insensitive matching
                        page_text_lower = page_text.lower()
                        
                        # Check if any of the keywords are in the page text
                        if any(keyword.lower() in page_text_lower for keyword in keywords):
                            logger.debug(f"Page {i+1} contains specified keywords, including text")
                            text += page_text
                        else:
                            logger.debug(f"Page {i+1} doesn't contain specified keywords, skipping")
                    else:
                        # In full mode or when no keywords are specified, include all text
                        text += page_text
                except Exception as e:
                    logger.error(f"Error processing page {i+1} of {file_path}: {str(e)}")
                    continue
        
        return text
    except FileNotFoundError:
        logger.error(f"PDF file not found: {file_path}")
        return ""
    except (pdfplumber.PDFSyntaxError, pdfplumber.pdfminer.pdfdocument.PDFEncryptionError) as e:
        logger.error(f"PDF parsing error for {file_path}: {str(e)}")
        return ""
    except MemoryError as e:
        logger.error(f"Memory error processing PDF {file_path} - file may be too large: {str(e)}")
        return ""
    except PermissionError as e:
        logger.error(f"Permission error accessing PDF {file_path}: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from PDF {file_path} using advanced method: {str(e)}")
        return ""


def extract_text_from_image(file_path: str) -> str:
    """
    Extract text from an image using OCR.
    
    Args:
        file_path (str): Path to the image file.
        
    Returns:
        str: Extracted text from the image.
    """
    try:
        # Open the image
        image = Image.open(file_path)
        
        # Apply common enhancement techniques
        image = enhance_image(image)
        
        # Perform OCR with optimized settings
        text = pytesseract.image_to_string(
            image,
            config='--psm 6 --oem 3'  # Page segmentation mode 6 (single block of text)
        )
        return text
    except FileNotFoundError:
        logger.error(f"Image file not found: {file_path}")
        return ""
    except IOError as e:
        logger.error(f"IO error opening image {file_path}: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from image {file_path}: {str(e)}")
        return ""


def advanced_extract_text_from_image(file_path: str, use_gpu: bool = None) -> str:
    """
    Extract text from an image using EasyOCR with preprocessing.
    
    Args:
        file_path (str): Path to the image file.
        use_gpu (bool, optional): Whether to use GPU for EasyOCR processing. If None, automatically
                                 detects GPU availability.
        
    Returns:
        str: Extracted text from the image.
    """
    try:
        # Open the image
        image = Image.open(file_path)
        
        # Preprocess the image
        image = preprocess_image(image)
        
        # Check if the image is mostly blank/empty before proceeding
        threshold = 230  # Threshold for considering a pixel as "white"
        white_ratio = np.sum(np.array(image) > threshold) / (image.width * image.height)
        
        # If image is more than 95% white/blank, skip OCR
        if white_ratio > 0.95:
            logger.info(f"Skipping mostly blank image: {file_path}")
            return ""
        
        # Convert image to numpy array
        image_np = np.array(image)
        
        # Auto-detect GPU if not specified
        if use_gpu is None:
            try:
                import torch
                use_gpu = torch.cuda.is_available()
                logger.debug(f"Auto-detected GPU availability: {use_gpu}")
            except ImportError:
                use_gpu = False
                logger.debug("PyTorch not available, defaulting to CPU mode")
        
        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'], gpu=use_gpu)
        
        # Perform OCR
        result = reader.readtext(image_np)
        
        # Concatenate text from result
        text = "\n".join([res[1] for res in result])
        return text
    except FileNotFoundError:
        logger.error(f"Image file not found: {file_path}")
        return ""
    except IOError as e:
        logger.error(f"IO error opening image {file_path}: {str(e)}")
        return ""
    except ValueError as e:
        logger.error(f"Value error processing image {file_path}: {str(e)}")
        return ""
    except MemoryError as e:
        logger.error(f"Memory error processing image {file_path} - file may be too large: {str(e)}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from image {file_path} using advanced method: {str(e)}")
        return ""


def extract_text_from_docx(file_path: str) -> str:
    """
    Extract text from all paragraphs in a DOCX file.
    
    Args:
        file_path (str): Path to the DOCX file.
        
    Returns:
        str: Extracted text from the DOCX.
    """
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs if paragraph.text])
        return text
    except FileNotFoundError:
        logger.error(f"DOCX file not found: {file_path}")
        return ""
    except docx.opc.exceptions.PackageNotFoundError:
        logger.error(f"Invalid DOCX file: {file_path}")
        return ""
    except Exception as e:
        logger.error(f"Error extracting text from DOCX {file_path}: {str(e)}")
        return ""


def detect_and_extract_text(file_path: str, advanced: bool = False, 
                           pdf_mode: str = "full", max_pages: int = 3,
                           keywords: Optional[List[str]] = None) -> str:
    """
    Detect file type and extract text using standard or advanced extraction methods.
    
    Args:
        file_path (str): Path to the file.
        advanced (bool): Whether to use advanced extraction methods.
        pdf_mode (str): For PDFs, extraction mode - "full" or "optimized".
        max_pages (int): Maximum pages to process in optimized mode.
        keywords (Optional[List[str]]): Keywords to filter text in optimized PDF mode.
        
    Returns:
        str: Extracted text from the file.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return ""
    
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        return advanced_extract_text_from_pdf(file_path, mode=pdf_mode, max_pages=max_pages, 
                                           keywords=keywords) if advanced else extract_text_from_pdf(file_path)
    elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']:
        return advanced_extract_text_from_image(file_path) if advanced else extract_text_from_image(file_path)
    elif file_extension in ['.docx', '.doc']:
        return extract_text_from_docx(file_path)
    else:
        logger.warning(f"Unsupported file format: {file_extension}")
        return ""


def extract_text(file_path: str) -> str:
    """
    Detect file type and call the appropriate extraction function.
    
    Args:
        file_path (str): Path to the file.
        
    Returns:
        str: Extracted text from the file.
    """
    return detect_and_extract_text(file_path, advanced=False)


def advanced_extract_text(file_path: str, pdf_mode: str = "full", 
                         max_pages: int = 3, keywords: Optional[List[str]] = None) -> str:
    """
    Detect file type and call the appropriate advanced extraction function.
    
    Args:
        file_path (str): Path to the file.
        pdf_mode (str): For PDFs, extraction mode - "full" or "optimized".
        max_pages (int): Maximum pages to process in optimized mode.
        keywords (Optional[List[str]]): Keywords to filter text in optimized PDF mode.
        
    Returns:
        str: Extracted text from the file.
    """
    return detect_and_extract_text(file_path, advanced=True, 
                                 pdf_mode=pdf_mode, max_pages=max_pages, keywords=keywords)

