# def read_new_lines(filename, last_position):
#     new_lines = []
    
#     with open(filename, 'r') as file:
#         file.seek(last_position)  # Move to the last read position
#         new_lines = file.readlines()  # Read all new lines
        
#         # Update the last_position to the current end of the file
#         last_position = file.tell()
    
#     return new_lines, last_position

# filename = 'gendock_project/celery.logs'
# last_position = 0  # You can initialize this to the last read position
# new_lines, last_position = read_new_lines(filename, last_position)

# for line in new_lines:
#     # Process each new line as needed
#     print(line)

dic = {"hi":[]}
print(dic[list(dic)[0]]==[])