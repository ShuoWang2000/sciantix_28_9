with open(f"optimization_{Talip1320.caseName}_{number_of_interval}.txt", 'w') as file:
    total_rows = len(results_data[0])
    for index, row in enumerate(results_data):
        line = "\t".join(row)  # Join columns with a tab separator
        if index == total_rows - 1:
            file.write(line)
        else:
            file.write(line + "\n")