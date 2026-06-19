import os
import re
import uuid
from typing import List, Dict, Any

from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table
from docx.document import Document as _Document
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl


class DocxChunkParser:

    def __init__(self):
        pass

    # -------------------------
    # PUBLIC API
    # -------------------------
    def parse(self, catalog: str, program: str, file_list: List[str] = []) -> List[Dict[str, Any]]:
        
        self.image_root = os.path.join('data', catalog, program, 'images')
        os.makedirs(self.image_root, exist_ok=True)

        self.chunks = []
        self.chunk_id = 0
        
        for file in file_list:

            self.header_stack = []
            self.current_chunk = None
            prev_was_heading = False
            
            file_path = os.path.join('data', catalog, program, file)
            doc = Document(file_path)
            blocks = list(self._iter_block_items(doc))

            i = 0
            while i < len(blocks):
                block = blocks[i]

                # -------------------------
                # PARAGRAPH
                # -------------------------
                if isinstance(block, Paragraph):
                    text = block.text.strip()
                    level = self._get_heading_level(block)

                    # -------- HEADING --------
                    if level is not None:
                        self._update_header_stack(level, text)

                        # H1/H2 закрывают текущий чанк
                        if level in (1, 2, 3):
                            self._save_chunk()

                        prev_was_heading = True
                        i += 1
                        continue

                    # -------- TABLE CAPTION HANDLING --------
                    # если текущий блок paragraph, а следующий — таблица, то
                    # этот paragraph считается caption
                    if (
                        text
                        and i + 1 < len(blocks)
                        and (isinstance(blocks[i + 1], Table) or self._get_blips(blocks[i + 1]))
                    ):
                        # caption для image или table обработаем на следующем шаге
                        i += 1
                        continue

                    # -------- IMAGE CAPTION HANDLING --------
                    # если paragraph содержит картинку
                    blips = self._get_blips(block)
                    if blips:
                        self._handle_image(block, blocks[i - 1], blips)
                        i += 1
                        continue

                    # -------- NORMAL TEXT --------
                    if text:
                        if prev_was_heading:
                            self._start_new_chunk(file_path, program)

                        if self.current_chunk:
                            self.current_chunk["text"] += self._normalize_sentence(text) + "\n"

                        prev_was_heading = False
                    i += 1
                    continue

                # -------------------------
                # TABLE
                # -------------------------
                elif isinstance(block, Table):
                    caption = self._get_previous_caption(blocks, i)

                    if not self.current_chunk:
                        i += 1
                        continue

                    rows = []
                    for row in block.rows:
                        rows.append([cell.text.strip() for cell in row.cells])

                    self.current_chunk["tables"].append({
                        "description": caption,
                        "data": rows
                    })

                    i += 1
                    continue

                i += 1

            self._save_chunk()
        return self.chunks

    # -------------------------
    # INTERNAL METHODS
    # -------------------------

    def _iter_block_items(self, parent):
        if isinstance(parent, _Document):
            parent_elm = parent.element.body
        else:
            parent_elm = parent._element

        for child in parent_elm.iterchildren():
            if isinstance(child, CT_P):
                yield Paragraph(child, parent)
            elif isinstance(child, CT_Tbl):
                yield Table(child, parent)


    def _get_heading_level(self, paragraph):
        if not paragraph.style:
            return None

        name = paragraph.style.name.lower()
        if "heading" in name:
            m = re.search(r"(\d+)", name)
            if m:
                return int(m.group(1))
        return None
    

    def _update_header_stack(self, level, text):
        while self.header_stack and self.header_stack[-1][0] >= level:
            self.header_stack.pop()
        self.header_stack.append((level, text))


    def _get_section(self):
        for level, text in reversed(self.header_stack):
            if level == 1:
                return self._normalize_sentence(text)
        return None
    

    def _get_subsection(self):
        for level, text in reversed(self.header_stack):
            if level == 2:
                return self._normalize_sentence(text)
        return None
    
    
    def _get_subsubsection(self):
        for level, text in reversed(self.header_stack):
            if level == 3:
                return self._normalize_sentence(text)
        return None
    

    def _start_new_chunk(self, file_path, program):
        self.current_chunk = {
            "id": self.chunk_id,
            "program": program,
            "section": self._get_section(),
            "subsection": self._get_subsection(),
            "subsubsection": self._get_subsubsection(),
            "text": "",
            "tables": [],
            "images": [],
            "source_file": file_path
        }
        self.chunk_id += 1


    def _save_chunk(self):
        if self.current_chunk:
            self.chunks.append(self.current_chunk)
            self.current_chunk = None


    def _get_blips(self, paragraph):
        if "blip" in paragraph._element.xml:
            return paragraph._element.xpath('.//*[local-name()="blip"]')
        return []
    

    def _handle_image(self, paragraph, prev_block, blips):
        if not self.current_chunk:
            return
        if isinstance(prev_block, Paragraph):
            if prev_block.text.strip():
                caption = self._normalize_sentence(prev_block.text)
            else:
                caption = ''
        else:
            caption = ''

        for blip in blips:
            rId = blip.get('{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed')
            image_part = paragraph.part.related_parts[rId]
            ext = image_part.filename.split(".")[-1]
            filename = f"{uuid.uuid4().hex}.{ext}"
        
            with open(os.path.join(self.image_root, filename), "wb") as f:
                f.write(image_part.blob)

            self.current_chunk["images"].append({
                "description": caption,
                "filename": filename
            })


    def _get_previous_caption(self, blocks, index):
        if index == 0:
            return None
        prev = blocks[index - 1]
        if isinstance(prev, Paragraph):
            if prev.text.strip():
                text = self._normalize_sentence(prev.text)
                if text:
                    return text
        return None
    
    
    def chunk_to_text(self, chunk: dict, prefix: str = '') -> str:
        parts = []

        # if chunk.get("section"):
        #     parts.append(chunk["section"])

        # if chunk.get("subsection"):
        #     parts.append(chunk["subsection"])

        if chunk.get("subsubsection"):
            parts.append(chunk["subsubsection"])
        elif chunk.get("subsection"):
            parts.append(chunk["subsection"])
        elif chunk.get("section"):
            parts.append(chunk["section"])

        if chunk.get("text"):
            parts.append(chunk["text"])

        # for table in chunk.get("tables", []):
        #     desc = table.get("description")
        #     if desc:
        #         parts.append(desc)

        # for img in chunk.get("images", []):
        #     desc = img.get("description")
        #     if desc:
        #         parts.append(desc)

        text = " ".join(parts)
        if prefix:
            prefix = f"{prefix}: "

        return f"{prefix + " ".join(text.split())}"
        
    
    def _normalize_sentence(self, text):
        text = " ".join(text.split())      
        if text:   
            if text.endswith(('.', '!', '?', ':', ';')):
                return text
            else:
                return text + '.'
        else:
            return text