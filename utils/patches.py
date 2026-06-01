import streamlit.elements.image as st_image

def apply_patches():
    """Applies necessary monkey-patches for compatibility."""
    if not hasattr(st_image, "image_to_url"):
        try:
            from streamlit.elements.lib.image_utils import image_to_url
            st_image.image_to_url = image_to_url
        except ImportError:
            pass
