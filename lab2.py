#exercitiul 1
import random
#facem functia de simulare(o simulare)
def simulate():
    urn = ["red"] * 3 + ["blue"] * 4 + ["black"] * 2 #facem urna

    die = random.randint(1, 6) #dam cu zarul
    if die in [2, 3, 5]: #daca iese numar prim,adaugam o bila neagra
        urn.append("black")
    elif die == 6: #daca iese 6,adaugam una rosie
        urn.append("red")
    else:
        urn.append("blue") #altfel,una albastra
    
    drawn = random.choice(urn) #extragem o bila
    return drawn == "red" #returnam 1 sau 0 in functie daca e intr adevar rosie sau nu

N=1_000_000 #rulam experimentul/functia de 1 milion de ori
count_red=sum(simulate() for _ in range(N))
estimated_prob=count_red/N

print(f"Probabilitatea de a alege o bila rosie: {estimated_prob:.4f}")


#1) bonus
#P(rosu)=P(nr prim)*P(rosu|nr prim)+P(6)*P(rosu|6)+P(restul)*P(rosu|restul) [formula]
#calcule:
#P(red)=(3/6)×(3/10)+(1/6)×(4/10)+(2/6)×(3/10)
#P(red)=(3/6)(0.3)+(1/6)(0.4)+(2/6)(0.3)
#P(red)=(0.5)(0.3)+(0.1667)(0.4)+(0.3333)(0.3)
#P(red)=0.15+0.0667+0.1=0.3167
#observam ca probabilitatea teoretica e aproape de cea care ne iese ruland(daca rulam experimentul de mai multe ori ne iese diferenta mai mica)

#exercitiul 2
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(321)

lambdas=[1,2,5,10]
n=1000

X1=np.random.poisson(1,n)
X2=np.random.poisson(2,n)
X3=np.random.poisson(5,n)
X4=np.random.poisson(10,n)

random_lambdas=np.random.choice(lambdas,size=n,replace=True)
X_random = np.array([np.random.poisson(lam) for lam in random_lambdas])

datasets = [X1, X2, X3, X4, X_random]
titles = ["Poisson(1)", "Poisson(2)", "Poisson(5)", "Poisson(10)", "Randomized"]

print("Media si variatia fiecarei distributii:")
for name, X in zip(titles,datasets):
    print(f"{name:12s}  Media = {np.mean(X):.2f}, Variatia = {np.var(X):.2f}")

fig, axs = plt.subplots(3, 2, figsize=(10, 8))
axs = axs.flatten()

for i, (data, title) in enumerate(zip(datasets,titles)):
    axs[i].hist(data, bins=range(0, max(data)+2), edgecolor='red')
    axs[i].set_title(title)
    axs[i].set_xlabel("Number of calls")
    axs[i].set_ylabel("Frequency")

plt.tight_layout()
plt.show()
