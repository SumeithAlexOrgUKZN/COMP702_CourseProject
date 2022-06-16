

fileName = "test.txt"

with open(fileName, "r") as f:
    lines = f.readlines()

row = ""
data = []
hits = []
answer = []

dictionary = {
    "'R010'" : 0,
    "'R020'" : 0,
    "'R050'" : 0,
    "'R100'" : 0,
    "'R200'" : 0
}

count = 0
for i in range(len(lines)):
    if (i != len(lines)-1):
        row = (lines[i])[:-1]
    else:
        row = (lines[i])

    if (len(row) == 0):
        continue

    data = row.split(" ::::: ")
    data[1] = data[1][ 1 : -1]
    # print(row)
    # print(data)

    # now can parse each line
    hits = data[1].split(", ")
    # print(hits)

    if (count % 11 == 0) and (count != 0):
        answer.append(dictionary)
        dictionary = {
            "'R010'" : 0,
            "'R020'" : 0,
            "'R050'" : 0,
            "'R100'" : 0,
            "'R200'" : 0
        }

    for k in range(len(hits)):
        dictionary[hits[k]] = dictionary[hits[k]] + 1

    count += 1
    # print(data[0])
    if (count == 53):
        print()

answer.append(dictionary)
# print(count)

for i in range(5):
    print(answer[i])


# dictionary = {
#             "'R010'" : 0,
#             "'R020'" : 0,
#             "'R050'" : 0,
#             "'R100'" : 0,
#             "'R200'" : 0
#         }

# print(dictionary["'R010'"])