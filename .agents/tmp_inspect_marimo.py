import marimo._code_mode as cm

async with cm.get_context() as ctx:
    print("cells", len(ctx.cells))
    for index, cell in enumerate(ctx.cells):
        code = cell.code.replace("\n", " ")[:260]
        print(index, cell.id, "name=", getattr(cell, "name", None), "status=", getattr(cell, "status", None), "code=", code)
