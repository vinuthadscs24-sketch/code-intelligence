import javalang
from typing import Dict, Any, Optional

class JavaASTParser:
    def parse_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                code = f.read()
            
            tree = javalang.parse.parse(code)
            return {"file_path": file_path, "tree": tree, "source_code": code}
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None

    def extract_symbols_and_relations(self, tree, source_code: str):
        methods_data = []

        for path, node in tree.filter(javalang.tree.ClassDeclaration):
            class_name = node.name
            implements_list = [impl.name for impl in node.implements] if node.implements else []

            for method in node.methods:
                method_name = method.name
                param_types = [p.type.name for p in method.parameters] if method.parameters else []
                param_str = ",".join(param_types)
                signature = f"{method_name}({param_str})"

                calls = set()
                instantiates = set()

                if method.body:
                    for _, sub_node in method.filter(javalang.tree.MethodInvocation):
                        calls.add(sub_node.member)
                    for _, sub_node in method.filter(javalang.tree.ClassInstanceCreation):
                        if hasattr(sub_node.type, "name"):
                            instantiates.add(sub_node.type.name)

                methods_data.append({
                    "class_name": class_name,
                    "method_name": method_name,
                    "signature": signature,
                    "implements": implements_list,
                    "relationships": {
                        "CALLS": sorted(list(calls)),
                        "INSTANTIATES": sorted(list(instantiates))
                    }
                })

        return methods_data
