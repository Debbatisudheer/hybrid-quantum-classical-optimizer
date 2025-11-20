def format_result(title, data):
    print("\n=== " + title + " ===")
    for k, v in data.items():
        print(f"{k}: {v}")
