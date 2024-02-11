with open("/Users/zoe/Documents/GitHub/zoematr/Reverse_Transformer/data/HU_sentences.txt", "r") as input_file, open("/Users/zoe/Documents/GitHub/zoematr/Reverse_Transformer/data/clean.txt", "w") as output_file:
    # Iterate over each line in the input file
    for line in input_file:
        # Remove unwanted characters (numbers, commas, and colons)
        cleaned_line = ''.join(char for char in line if char.isalpha() or char.isspace())
        # Write the cleaned line to the output file
        output_file.write(cleaned_line)