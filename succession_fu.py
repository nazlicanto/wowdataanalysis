import re

# Define the function with the file argument
def count_fuckwords(finalescript):
    counts = {}
    char = None

    # Create a dictionary for the character names to standardize 
    character_name = {
        'KENDALL ROY': 'KENDALL',
        'SHIV ROY': 'SHIV',
        'ROMAN ROY': 'ROMAN',
        'LUKAS MATSSON': 'LUKAS',
        'HUGO BAKER': 'HUGO',
        'OSKAR GUDJOHNSON': 'OSKAR',
        'PETER MUNION': 'PETER',
        'FRANK VERNON': 'FRANK'
    }

    with open(finalescript, 'r') as file:
        lines = file.readlines()

    # The pattern for seperating the character name from the script
    char_name_pattern = r'^- ([A-Z\s]+)\:'


    for line in lines:
        # Try to find a character's name at the start of the line
        match = re.match(char_name_pattern, line)
        if match:
            char = match.group(1)
            # If the character's name is in the dictionary, replace it
            if char in character_name:
                char = character_name[char]
            # If the character's name is not in the dictionary, add it
            if char not in counts:
                counts[char] = 0
        # Only count "fuck" if char != 0
        if char is not None:
            # Count "fuck" in the line 
            f_count = line.lower().count('fuck')
            counts[char] += f_count

    return counts

filename = '...DATA//HI_English.srt'
print(count_fwords(finalescript))