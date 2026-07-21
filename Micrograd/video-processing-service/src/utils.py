extension_map = {
    ".mp4": "mp4v",   
    ".avi": "XVID",   
    ".webm": "VP80",  
    ".mov": "mp4v",   
    ".mkv": "X264",   
}

def get_fourcc_from_suffix(suffix):
    return extension_map[suffix]