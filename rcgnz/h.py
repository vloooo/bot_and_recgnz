for i in range(50):
    if i < 10:
        k = chr(i + 48)  # get character from results
    else:
        k = chr(i + 55)
    print(i, "                  ", k)