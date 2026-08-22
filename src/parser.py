import javalang

class JavaASTParser:
    def parse_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            source_code = f.read()
        tree = javalang.parse.parse(source_code)
        return tree, source_code

    def extract_symbols_and_relations(self, tree, source_code):
        methods = []
        classes = []

        if tree is None:
            return {"methods": methods, "classes": classes}

        # Extract Class Declarations
        for _, class_decl in tree.filter(javalang.tree.ClassDeclaration):
            classes.append(class_decl.name)

        # Extract Method Declarations
        for _, method in tree.filter(javalang.tree.MethodDeclaration):
            calls = []
            instantiations = []

            # Method Invocations
            for _, call in method.filter(javalang.tree.MethodInvocation):
                if call.member:
                    calls.append(call.member)

            # Object Instantiations -> Use ClassCreator
            for _, creator in method.filter(javalang.tree.ClassCreator):
                if hasattr(creator, "type") and hasattr(creator.type, "name"):
                    instantiations.append(creator.type.name)

            annotations = [ann.name for ann in getattr(method, "annotations", [])]

            methods.append({
                "name": method.name,
                "enclosing_class": getattr(method, "_enclosing_class", "Global"),
                "annotations": annotations,
                "calls": calls,
                "instantiations": instantiations,
                "source_code": source_code  # Fallback to full source if range slice is omitted
            })

        return {
            "classes": classes,
            "methods": methods
        }