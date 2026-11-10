"""Image formats shared by extraction, inventories, training and evaluation."""
from PIL import Image

FORMAT_EXTENSIONS = {"JPEG": ".JPEG", "PNG": ".png", "TIFF": ".tiff", "BMP": ".bmp", "WEBP": ".webp"}
EXTENSIONS = frozenset({v.lower() for v in FORMAT_EXTENSIONS.values()} | {".jpg", ".tif"})


def inspect_image(source, *, context):
    """Fully decode and validate the RGB read path without re-encoding source bytes."""
    format_name = "unidentified"
    try:
        with Image.open(source) as image:
            format_name = image.format
            if format_name not in FORMAT_EXTENSIONS:
                raise ValueError("Unsupported source format")
            image.load()
            with image.convert("RGB") as rgb:
                rgb.load()
            return {"format": format_name, "mode": image.mode, "size": list(image.size),
                    "extension": FORMAT_EXTENSIONS[format_name]}
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError(f"{context}; format={format_name}: {error}") from error
