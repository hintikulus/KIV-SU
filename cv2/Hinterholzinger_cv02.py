import matplotlib.pyplot as plt
import numpy as np

POINTS = 3
def main():

    # Inicializace vektoru
    x = np.zeros(POINTS)
    y = np.zeros(POINTS)

    # Nacteni bodu z konzole
    for i in range(0, POINTS):
        print("Zadejte", str(i+1) + ".", "bod ve formatu: x;y")
        input_data = input().split(';', 1)
        x[i] = float(input_data[0])
        y[i] = float(input_data[1])

    # Testovaci data ze zadani
    #x = np.array([1, 5, 2])
    #y = np.array([1, 1, 4])

    # Vykresleni trojuhelniku
    for i in range(0, POINTS):
        plt.plot(np.roll(x, i)[:2], np.roll(y, i)[:2], "orange")

    # Vypocitani souradnic stredu stran
    s_x = (x + np.roll(x, -1)) * .5
    s_y = (y + np.roll(y, -1)) * .5

    # Vykresleni stredu stran
    for i in range(0, POINTS):
        plt.plot(s_x[i], s_y[i], "rx")

    # Souradnice smeroveho vektoru
    v_x = x - np.roll(x, -1)
    v_y = y - np.roll(y, -1)

    # Matice souradnic smerovych vektoru
    dir_v = np.array([v_x, v_y]).transpose()

    # Vektor skalaru souradnic vektoru a stredu stran
    scalars = v_x * s_x + v_y * s_y

    # Vypocet stredu kruznice opsane (ze dvou stran)
    S = np.linalg.inv(dir_v[:2]).dot(scalars[:2])

    # Vykresleni stredu kruznice opsane
    plt.plot(S[0], S[1], "rx")

    # Vypocet vzdalenosti stredu kruznice od vrcholu (vsech)
    r = ((x - S[0])**2 + (y - S[1])**2)**.5

    # Vykresleni kruznice
    circle = plt.Circle((S[0], S[1]), r[0], color = 'b', fill = False)
    plt.gca().add_patch(circle)

    plt.show()

if __name__ == '__main__':
    main()
