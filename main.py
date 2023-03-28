import numpy as np
import matplotlib.pyplot as plt

P = 250
n = 0.8
Tmax = 1000

city = 1

# City 4
if city == 4:
    x = [3, 2, 12, 7,  9,  3, 16, 11, 9, 2]
    y = [1, 4, 2, 4.5, 9, 1.5, 11, 8, 10, 7]
# City 3
elif city == 3:
    x = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
    y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
# City 2
elif city == 2:
    x = [0, 2, 6, 7, 15, 12, 14, 9.5, 7.5, 0.5]
    y = [1, 3, 5, 2.5, -0.5, 3.5, 10, 7.5, 9, 10]
# City 1
elif city == 1:
    x = [0, 3, 6, 7, 15, 12, 14, 9, 7, 0]
    y = [1, 4, 5, 3, 0, 4, 10, 6, 9, 10]
else:
    print("There is not such city")

cities = np.array([i for i in range(len(x))])
city_length = len(cities)

# Computing distance between all cities
def distance_matrix(x, y):

    s = (len(x),len(y))
    distance = np.zeros(s)

    for i in range(len(x)):
        for j in range(len(y)):
            distance[i][j] = ((x[i]-x[j])**2+(y[i]-y[j])**2)**0.5

    return np.array(distance)

# Creating first population set
def initialPopulation(cities, n_population):
    population_set = []
    for i in range(n_population):
        rand = cities[np.random.choice(cities, len(cities), replace=False)]
        population_set.append(rand)
    return np.array(population_set)

# Calculating total distances
def total_distance(distances, population):
    list = []
    for i in range(0, len(population)):
        temp = 0
        for j in range(0, len(cities)-1):
            temp += distances[population[i][j], population[i][j+1]]
        temp += distances[population[i][-1], population[i][0]]
        list.append(temp)
    return np.array(list)

# Rulette drawing two parents
def selection_operator(distances, population):

    output = []

    total_distances = distances.sum()
    probability_list = distances/total_distances

    for i in range(0, int(n*P)):
        parent = np.random.choice(population.shape[0], p=probability_list)
        output.append(population[parent])
    np.random.shuffle(output)
    return np.array(output)

def cross_over(population):
    child = []

    for i in range(0, len(population)-1, 2):
        parent1 = population[i]
        parent2 = population[i+1]

        # Creating copy of parents
        parent1_copy = parent1.tolist()
        parent2_copy = parent2.tolist()

        # Length of chromosome
        chrom_length = len(parent1)

        # Initialization of childrens
        child1 = np.array([-1] * chrom_length)
        child2 = np.array([-1] * chrom_length)

        swap = True
        pos = 0

        if swap == True:
            while True:
                child1[pos] = parent1[pos]
                pos = parent2.tolist().index(parent1[pos])
                if parent1_copy[pos] == -1:
                    swap = False
                    break
                parent1_copy[pos] = -1

        elif swap == False:
            while True:
                child1[pos] = parent2[pos]
                pos = parent1.tolist().index(parent2[pos])
                if parent2_copy[pos] == -1:
                    swap = True
                    break
                parent2_copy[pos] = -1

        for i in range(chrom_length):
            if child1[i] == parent1[i]:
                child2[i] = parent2[i]
            else:
                child2[i] = parent1[i]

            if child2[i] == parent2[i]:
                child1[i] = parent1[i]
            else:
                child1[i] = parent2[i]

        child1_list = child1.tolist()
        child.append(child1_list)
        child2_list = child2.tolist()
        child.append(child2_list)

    return np.array(child)

def mutation(offsprings):

    for i in range(0, len(offsprings)):
        a = np.random.randint(0, city_length)
        b = np.random.randint(0, city_length)

        offsprings[i][a], offsprings[i][b] = offsprings[i][b], offsprings[i][a]
    return offsprings

def TheBestPopulation(population, distances):
    sort = np.argsort(distances)
    best_population = population[sort]
    return best_population

def TheBestDistance(distances):
    sort = np.argsort(distances)
    return distances[sort][0]

# 1. Distance of cities
distance = distance_matrix(x,y)


# 2. Initial population
initial_population = initialPopulation(cities, P)

# loop

best_sum_distance = []


for i in range(0, Tmax):

    # 3. Total distance of all population
    sum_distance = total_distance(distance, initial_population)

    best_distance = TheBestDistance(sum_distance)
    best_sum_distance.append(best_distance)

    # 4. Randomy selection of offsprings - Rulette selection
    prob_list = selection_operator(sum_distance, initial_population)

    # 5. CX

    CX = cross_over(prob_list)

    # 6. Mutation
    final_childrens = mutation(CX)


    sum_distance_children = total_distance(distance, final_childrens)


    bestparents = TheBestPopulation(initial_population, sum_distance)


    bestchildren = TheBestPopulation(final_childrens, sum_distance_children)


    initial_population = np.concatenate((bestparents[:int(P-(n*P))], bestchildren[:int(n*P)]))

optimum_distance = min(best_sum_distance)

print("Distance: ", optimum_distance)
best_city_combination = initial_population[0]
best_city_list = best_city_combination.tolist()
best_city_list.append(best_city_list[0])
print("Best city combination: ",best_city_list)

plt.scatter(x, y)
for i in range(0, len(best_city_list)-1):
    x_values = [x[best_city_list[i]], x[best_city_list[i+1]]]
    y_values = [y[best_city_list[i]], y[best_city_list[i+1]]]
    plt.plot(x_values, y_values, 'bo', linestyle="--")

for i in range(0, city_length):
    plt.text(x[cities[i]] - 0.015, y[cities[i]] + 0.25, i)

plt.title('Distance: {:.2f}, city order: {}'.format(optimum_distance, best_city_list))
plt.xlabel("x")
plt.ylabel("y")

plt.show()
