#convert.py

import xml.etree.ElementTree as ET
from xml.etree.ElementTree import tostring, tostringlist
import pandas as pd
import numpy as np
import os

def get_original_text(element):
    """Extract the raw text of a turn exactly as it appears in the XML file, including all tags"""
    # Get the element's inner text content including children tags
    content = ''.join([el for el in element.itertext()]).strip()
    return content

def subel_to_string(subel, just_value = False):
    if subel.tag == 'turn':
        content = subel.text
        return content if content is not None else ''
    k, v = list(subel.attrib.items())[0]
    if not just_value:
        content = f'<{subel.tag} {k}=\"{v}\">{subel.text}</{subel.tag}>'
        return content
    else:
        return v

def get_text(element, just_value = False):
    content_list = []
    for subel in element.iter():
        subel_string = subel_to_string(subel, just_value=just_value)
        if subel.tail:
            tail = subel.tail
            subel_string += f'{tail}'
            subel_string = subel_string.strip()
        if subel_string:
            content_list.append(subel_string)
    content = ' '.join(content_list)
    return content

def get_inner_xml(element):
    """Preserve tags and their attributes inside an element."""
    print(element.attrib)
    out = []
    for e in element:
        out.append(ET.tostring(e, encoding="unicode"))
    out_string = ''.join(out)
    return out_string

def main():
    pd.set_option('display.max_rows', 100)
    # pd.set_option('display.max_colwidth', 75)
    # Parse XML file
    for root, dirs, files in os.walk('./data'):
        for F in files:
            if F.endswith('.era'):
                filename = os.path.join(root, F)
                tree = ET.parse(filename)
                root = tree.getroot()

                # Extract <text> metadata
                elem = root.find(".//text")
                meta = elem.attrib
                meta_df = pd.DataFrame([meta])

                # Extract annotated turns
                turns = []
                for turn in elem.findall(".//turn"):
                    speaker = turn.attrib.get("who", "student")
                    turn_type = turn.attrib.get("type", "")
                    
                    text_og = get_original_text(turn)
                    text_an = get_text(turn, just_value = False)
                    text_ok = get_text(turn, just_value = True)
                    
                    turns.append({
                        "speaker": speaker,
                        "turn_type": turn_type,
                        "text_an": text_an,
                        "text_og": text_og,
                        "text_ok": text_ok,
                    })
                    ...

                # Create conversation DataFrame
                turns_df = pd.DataFrame(turns)
                turns_df['text_an'] = turns_df['text_an'].apply(lambda x: x if x else np.nan)
                # turns_df = turns_df.dropna()
                print(turns_df)

if __name__ == "__main__":
    main()