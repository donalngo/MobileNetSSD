def generator(batch_size, data):
    step=0
    while (step+1)* batch_size < len(data):
        yield data[step*batch_size: (step+1)*batch_size]
        step+=1
    yield data[step*batch_size:]


num_gen = generator(3, range(14))

for nums in num_gen:
    print(nums)