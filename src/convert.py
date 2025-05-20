#convert.py

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np

pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

def get_full_turn_text(element):
    """Extract the raw text of a turn exactly as it appears in the XML file, including all tags"""
    # Get the element's inner text content including children tags
    content = ""
    if element.text:
        content += element.text
    
    for child in element:
        # Convert child element to string with its tags and content
        child_str = ET.tostring(child, encoding="unicode")
        content += child_str
        
        # Add any tail text that appears after the child
        if child.tail:
            content += child.tail
    
    return content

def get_inner_xml(element):
    """Preserve tags and their attributes inside an element."""
    print(element.attrib)
    out = []
    for e in element:
        out.append(ET.tostring(e, encoding="unicode"))
    out_string = ''.join(out)
    return out_string

# Parse XML file
tree = ET.parse("./data/unibo_1.era")
root = tree.getroot()

# Extract <text> metadata
elem = root.find(".//text")
meta = elem.attrib
meta_df = pd.DataFrame([meta])

# Extract annotated turns
turns = []
for turn in elem.findall(".//turn"):
    speaker = turn.attrib.get("who", "student")  # "student" if not chatbot
    turn_type = turn.attrib.get("type", "")
    
    text_og = "".join(turn.itertext()).strip()
    text_an = get_full_turn_text(turn)

    turns.append({
        "speaker": speaker,
        "turn_type": turn_type,
        "text_an": text_an,
        "text_og": text_og,
        # "correct_text": correct_text,
    })

# Create conversation DataFrame
turns_df = pd.DataFrame(turns)
turns_df['text_an'] = turns_df['text_an'].apply(lambda x: x if x else np.nan)
# turns_df = turns_df.dropna()
print(turns_df)