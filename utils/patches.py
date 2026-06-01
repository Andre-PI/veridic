import streamlit.elements.image as st_image

def apply_patches():
    """Applies necessary monkey-patches for compatibility."""
    
    # 1. Locate original image_to_url function
    image_to_url_fn = None
    if hasattr(st_image, "image_to_url"):
        image_to_url_fn = st_image.image_to_url
    else:
        try:
            from streamlit.elements.lib.image_utils import image_to_url
            image_to_url_fn = image_to_url
        except ImportError:
            pass

    if image_to_url_fn is None:
        return

    # 2. Define a wrapper that adapts older 'int' width parameter to newer 'LayoutConfig' object
    class DummyLayoutConfig:
        def __init__(self, width):
            self.width = width

    original_fn = image_to_url_fn

    def patched_image_to_url(image, layout_config, *args, **kwargs):
        # If older code passes a simple int/float width instead of LayoutConfig
        if isinstance(layout_config, (int, float)):
            layout_config = DummyLayoutConfig(layout_config)
        return original_fn(image, layout_config, *args, **kwargs)

    st_image.image_to_url = patched_image_to_url
