import pytest
import tempfile
import os
from src.utils import parse_story_xml


class TestParseStoryXml:
    """Unit tests for the parse_story_xml function."""
    
    def test_parse_xml_with_namespace(self):
        """Test parsing XML with a default namespace (like the story.xml file)."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<story xmlns="urn:whodunit:sms:1">
  <chapter id="1">
    <message id="m1">
      <sender ref="alice"/>
      <receiver ref="bob"/>
      <body>Hello Bob!</body>
    </message>
    <message id="m2">
      <sender ref="bob"/>
      <receiver ref="alice"/>
      <body>Hi Alice!</body>
    </message>
  </chapter>
</story>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            messages = parse_story_xml(temp_path)
            
            assert len(messages) == 2
            assert messages[0]['sender'] == 'alice'
            assert messages[0]['receiver'] == 'bob'
            assert messages[0]['body'] == 'Hello Bob!'
            assert messages[1]['sender'] == 'bob'
            assert messages[1]['receiver'] == 'alice'
            assert messages[1]['body'] == 'Hi Alice!'
        finally:
            os.unlink(temp_path)
    
    def test_parse_xml_without_namespace(self):
        """Test parsing XML without a namespace."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<story>
  <message id="m1">
    <sender ref="charlie"/>
    <receiver ref="dave"/>
    <body>Test message</body>
  </message>
</story>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            messages = parse_story_xml(temp_path)
            
            assert len(messages) == 1
            assert messages[0]['sender'] == 'charlie'
            assert messages[0]['receiver'] == 'dave'
            assert messages[0]['body'] == 'Test message'
        finally:
            os.unlink(temp_path)
    
    def test_parse_nested_messages_in_chapters(self):
        """Test parsing messages nested inside chapter elements."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<story xmlns="urn:whodunit:sms:1">
  <chapter id="1" title="Chapter One">
    <message id="m1">
      <sender ref="alice"/>
      <receiver ref="bob"/>
      <body>Chapter 1 message</body>
    </message>
  </chapter>
  <chapter id="2" title="Chapter Two">
    <message id="m2">
      <sender ref="bob"/>
      <receiver ref="alice"/>
      <body>Chapter 2 message</body>
    </message>
  </chapter>
</story>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            messages = parse_story_xml(temp_path)
            
            assert len(messages) == 2
            assert messages[0]['body'] == 'Chapter 1 message'
            assert messages[1]['body'] == 'Chapter 2 message'
        finally:
            os.unlink(temp_path)
    
    def test_full_text_format(self):
        """Test that full_text field is formatted correctly."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<story xmlns="urn:whodunit:sms:1">
  <message id="m1">
    <sender ref="alice"/>
    <receiver ref="bob"/>
    <body>Hello!</body>
  </message>
</story>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            messages = parse_story_xml(temp_path)
            
            expected_full_text = (
                '<sender ref="alice"/>\n'
                '<receiver ref="bob"/>\n'
                '<body>Hello!</body>'
            )
            assert messages[0]['full_text'] == expected_full_text
        finally:
            os.unlink(temp_path)
    
    def test_empty_body(self):
        """Test parsing message with empty body."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<story xmlns="urn:whodunit:sms:1">
  <message id="m1">
    <sender ref="alice"/>
    <receiver ref="bob"/>
    <body></body>
  </message>
</story>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            messages = parse_story_xml(temp_path)
            
            assert len(messages) == 1
            assert messages[0]['body'] == ''
        finally:
            os.unlink(temp_path)
    
    def test_missing_sender_element(self):
        """Test parsing message with missing sender element."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<story xmlns="urn:whodunit:sms:1">
  <message id="m1">
    <receiver ref="bob"/>
    <body>No sender!</body>
  </message>
</story>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            messages = parse_story_xml(temp_path)
            
            assert len(messages) == 1
            assert messages[0]['sender'] == 'Unknown'
            assert messages[0]['receiver'] == 'bob'
        finally:
            os.unlink(temp_path)
    
    def test_no_messages(self):
        """Test parsing XML with no messages returns empty list."""
        xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<story xmlns="urn:whodunit:sms:1">
  <participants>
    <person id="alice"/>
  </participants>
</story>'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False) as f:
            f.write(xml_content)
            temp_path = f.name
        
        try:
            messages = parse_story_xml(temp_path)
            assert len(messages) == 0
        finally:
            os.unlink(temp_path)
    
    def test_parse_actual_story_file(self):
        """Integration test: parse the actual story.xml file."""
        story_path = "data/story.xml"
        
        if not os.path.exists(story_path):
            pytest.skip("data/story.xml not found")
        
        messages = parse_story_xml(story_path)
        
        # Verify we got messages
        assert len(messages) > 0, "Should parse at least some messages from story.xml"
        
        # Verify structure of first message
        first_msg = messages[0]
        assert 'sender' in first_msg
        assert 'receiver' in first_msg
        assert 'body' in first_msg
        assert 'full_text' in first_msg
        
        # Verify sender/receiver are not Unknown (i.e., parsing worked)
        assert first_msg['sender'] != 'Unknown'
        assert first_msg['receiver'] != 'Unknown'
        
        print(f"\n✓ Successfully parsed {len(messages)} messages from story.xml")
        print(f"  First message: {first_msg['sender']} -> {first_msg['receiver']}")
