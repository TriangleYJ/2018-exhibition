from gd_deep_q_learning import geometry

if __name__ == "__main__":
    G = geometry.geometry()
    while True:
        D = G.frame_step([1,0])
        print(D[1])