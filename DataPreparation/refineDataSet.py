import os

# assign directory
directoryhq = 'ROIedHQ/'
directorylq = 'ROIedLQ/'
 
# iterate over files in
# that directory
fileslq = []
for filename in os.listdir(directorylq):
    flq = os.path.join(directorylq, filename)
    if '(' in filename:
        os.remove(flq)

    else:
        fileslq.append(filename)


for filename in os.listdir(directoryhq):
    fhq = os.path.join(directoryhq, filename)
    if filename not in fileslq:
        os.remove(fhq)

print(len(fileslq))

