# def read_new_lines(filename, last_position):
#     new_lines = []
    
#     with open(filename, 'r') as file:
#         file.seek(last_position)  # Move to the last read position
#         new_lines = file.readlines()  # Read all new lines
        
#         # Update the last_position to the current end of the file
l1 = ['ads','b1','c4']
l2 = ['x','y','z']
r = list(range(len(l1)))
# enumerate(zip(l1,l2))
a,b,c = zip(l1,l2,r)
print(a)