import xml.etree.ElementTree as ET
from typing import List, Dict

def parse_story_xml(file_path: str) -> List[Dict[str, str]]:
    """
    Parses the story XML file into a list of message dictionaries.
    
    Assumed structure:
    <story>
        <message>
            <sender ref="..."/>
            <receiver ref="..."/>
            <body>...</body>
        </message>
    </story>
    """
    # In a real scenario, handle FileNotFoundError here
    tree = ET.parse(file_path)
    root = tree.getroot()
    
    # Get the namespace from the root element's tag
    # ElementTree parses default namespace into Clark notation: {namespace}tagname
    namespace = {}
    if root.tag.startswith('{'):
        ns_uri = root.tag.split('}')[0][1:]  # Extract namespace URI
        namespace = {'ns': ns_uri}
    
    messages = []
    
    # Messages can be directly under root or inside chapter elements
    # Use namespace-aware search with .// to search recursively
    if namespace:
        # Search for messages with namespace (recursively with .//)
        for msg in root.findall('.//ns:message', namespace):
            sender_elem = msg.find('ns:sender', namespace)
            receiver_elem = msg.find('ns:receiver', namespace)
            body_elem = msg.find('ns:body', namespace)
            
            sender = sender_elem.get('ref') if sender_elem is not None else "Unknown"
            receiver = receiver_elem.get('ref') if receiver_elem is not None else "Unknown"
            body = body_elem.text if body_elem is not None and body_elem.text else ""
            
            # We format the full text block here. This is what the LLM will see.
            formatted_text = (
                f"<sender ref=\"{sender}\"/>\n"
                f"<receiver ref=\"{receiver}\"/>\n"
                f"<body>{body}</body>"
            )
            
            messages.append({
                "sender": sender,
                "receiver": receiver,
                "body": body,
                "full_text": formatted_text.strip()
            })
    else:
        # Fallback: try without namespace (for XML files without namespaces)
        for msg in root.findall('.//message'):
            sender = msg.find('sender').get('ref') if msg.find('sender') is not None else "Unknown"
            receiver = msg.find('receiver').get('ref') if msg.find('receiver') is not None else "Unknown"
            body = msg.find('body').text if msg.find('body') is not None else ""
            
            # We format the full text block here. This is what the LLM will see.
            formatted_text = (
                f"<sender ref=\"{sender}\"/>\n"
                f"<receiver ref=\"{receiver}\"/>\n"
                f"<body>{body}</body>"
            )
            
            messages.append({
                "sender": sender,
                "receiver": receiver,
                "body": body,
                "full_text": formatted_text.strip()
            })
    
    return messages