from pathlib import Path
import tree_sitter_java as tsjava
from tree_sitter import Language, Parser, Tree, Node

class JavaASTParser:
    def __init__(self):
        # Initialize Java language binding for Tree-sitter
        self.JAVA_LANGUAGE = Language(tsjava.language())
        self.parser = Parser(self.JAVA_LANGUAGE)

    def parse_file(self, file_path: Path) -> tuple[Tree, bytes]:
        """Reads a Java file and returns its Concrete Syntax Tree along with source bytes."""
        with open(file_path, "rb") as f:
            source_code = f.read()
        
        tree = self.parser.parse(source_code)
        return tree, source_code

    @staticmethod
    def get_node_text(node: Node, source_code: bytes) -> str:
        """Extracts the exact source code string corresponding to an AST node."""
        return source_code[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    @staticmethod
    def find_children_of_type(node: Node, type_name: str) -> list[Node]:
        """Utility to find all direct children of a specific syntax node type."""
        return [child for child in node.children if child.type == type_name]