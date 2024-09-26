import os
import subprocess


def latex_bf(s):
    return r"\textbf{" + s + r"}"


def latex_color(s, color="red"):
    return r"\textcolor{" + color + r"}{" + s + r"}"


def latex_format(s, use_latex=False, **kwargs):
    if use_latex:
        if kwargs.get("bf", False):
            s = latex_bf(s)
        if kwargs.get("color", False):
            s = latex_color(s, kwargs["color"])
    return s


def savefig(fig, filename, *args, **kwargs):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)

    fig.savefig(filename, bbox_inches="tight", *args, **kwargs)
    gs_opt(filename)


def gs_opt(filename):
    filename_tmp = filename.split(".")[-2] + "_tmp.pdf"
    gs = [
        "gs",
        "-sDEVICE=pdfwrite",
        "-dEmbedAllFonts=true",
        "-dSubsetFonts=false",  # Create font subsets (default)
        "-dPDFSETTINGS=/ebook",  # Image resolution
        "-dAutoRotatePages=/None",  # Rotation
        "-dDetectDuplicateImages=true",  # Embeds images used multiple times only once
        "-dCompressFonts=false",  # Compress fonts in the output (default)
        "-dNOPAUSE",  # No pause after each image
        "-dQUIET",  # Suppress output
        "-dBATCH",  # Automatically exit
        "-sOutputFile=" + filename_tmp,  # Save to temporary output
        filename,
    ]  # Input file

    subprocess.run(gs)  # Create temporary file
    #     subprocess.run(['rm', filename])            # Delete input file
    subprocess.run(["mv", filename_tmp, filename])  # Rename temporary to input file
