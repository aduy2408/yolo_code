import marimo._code_mode as cm

async with cm.get_context() as ctx:
    for key in ["aRyZ", "MUjO", "btOL", "YdPm", "xfuM"]:
        cell = ctx.cells[key]
        print(f"\n===== CELL {key} =====")
        print(cell.code)
