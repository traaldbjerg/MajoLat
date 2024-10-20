import majorization as mj
import entropy as en

a = mj.ProbVector([0.6, 0.15, 0.15, 0.1])
b = mj.ProbVector([0.5, 0.25, 0.2, 0.05])

print("\nMajorization calculations:\n")

print(a)
print(b)
print(a.majorizes(b))
print(b.majorizes(a))

meet = a + b
join = a * b

print(meet)
print(join)

print(meet > a)
print(meet > b)
print(meet < a)
print(meet < b)
print(join > a)
print(join > b)
print(join < a)
print(join < b)

print("\nEntropy calculations:\n")

H_a = en.shannon_entropy(a)
H_b = en.shannon_entropy(b)
H_meet = en.shannon_entropy(meet)
H_join = en.shannon_entropy(join)

print(H_a)
print(H_b)
print(H_meet)
print(H_join)
#supermodularity test
print(H_meet + H_join >= H_a + H_b)
#subadditivity test
print(H_meet <= H_a + H_b)

