# Reinforcement Learning pada Game Flappy Bird

## Anggota Kelompok:

| Name                      | NRP        | Class                  |
| ------------------------- | ---------- | ---------------------- |
| Bunga Melati Putri Luqman | 5025231253 | Pembelajaran Mesin (D) |
| Andi Nur Nabila Syalwani  | 5025231104 | Pembelajaran Mesin (D) |
| Alma Khusnia              | 5025231063 | Pembelajaran Mesin (D) |
| Alya Rahmatillah Machmud  | 5025231315 | Pembelajaran Mesin (D) |

## Overview

Penugasan ini mengacu pada repositori yenchenlin/DeepLearningFlappyBird, yang menggunakan algoritma Deep Reinforcement Learning berbasis Deep Q-Network (DQN) dengan bantuan TensorFlow dan OpenCV. Namun, dalam tugas Hands-On modul 5 ini, pendekatan tersebut disederhanakan dengan mengganti algoritma DQN menjadi Q-Learning.

Agen yakni burung dikendalikan oleh algoritma Q-learning untuk mencapai goal yaitu menghindari tabrakan dengan pipa dan memaksimalkan skor. Aksi yang dilakukan ada 2 yaitu 0 (tidak melakukan apa-apa) artinya burung jatuh karena gravitasi dan 1 (terbang/flap) dengan burung terbang keatas untuk menghindari pipa

![image](https://github.com/user-attachments/assets/32ab4609-4c97-4fc0-9b31-bb58247edbf1)

## Install Dependencies:

- Python 2.7 or 3
- pygame

## Cara Menjalankan:

```
git clone https://github.com/nabilasyalwani/reinforcement-learning-flappy-bird.git
cd reinforcement-learning-flappy-bird
python q_learning.py
```

## Penjelasan:

Q-Learning adalah salah satu algoritma dalam RL yang termasuk dalam kategori Value-Based. Algoritma ini mempelajari fungsi nilai aksi Q ( s , a ) untuk memaksimalkan reward yang diperoleh Agen.

**Value-Based**: Menentukan policy secara tidak langsung dengan **mempelajari fungsi nilai aksi** $Q(s, a)$ dan memilih aksi dengan nilai tertinggi.

$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
$$

- **$Q(s, a)$** = Seberapa baik Agen dalam mengambil aksi ($a$) pada suatu state ($s$).
- **$\alpha$** = _Learning rate_ pada fungsi nilai.
- **$r$** = Reward yang diterima setelah melakukan aksi ($a$).
- **$\gamma$** = _Discount factor_ yang menentukan seberapa jauh Agen mempertimbangkan reward masa depan.
- **$\max_{a'} Q(s', a')$** = Nilai terbaik yang bisa diperoleh dari state selanjutnya ($s'$).

```
# Hyperparameter
ALPHA = 0.7
GAMMA = 0.95
INITIAL_EPSILON = 0.01
EXPLORE = 100000
FINAL_EPSILON = 0.01
EPISODES = 10000
```

Bagian ini mengatur nilai-nilai yang memengaruhi cara agen belajar dan seberapa banyak eksplorasi dilakukan.
<br>

```
ACTIONS = 2  # 0: do nothing, 1: flap
```

Burung hanya bisa melakukan dua aksi: 0 (diam) dan 1 (terbang).
<br>

```
# Fungsi diskretisasi
def get_features(game_state):
    bird_y = game_state.get_bird_y()
    pipe_x = game_state.get_next_pipe_x()
    pipe_gap_y = game_state.get_next_pipe_gap_y()
    return (bird_y, pipe_x, pipe_gap_y)
```

Fungsi ini mengambil informasi penting dari file `game`, yakni posisi burung, posisi pipa terdekat, dan posisi celah pipa dalam nilai diskrit sehingga skala untuk statenya tidak terlalu besar dan masih dapat ditampung oleh Q-table.
<br>

```
# Inisialisasi Q-table
Q = np.zeros((NUM_BINS, NUM_BINS, NUM_BINS, ACTIONS))
```

Bagian ini membuat Q-table kosong untuk menyimpan nilai Q dari kombinasi state (diskretisasi 3 variabel) dan 2 aksi.
<br>

```
def choose_action(state, epsilon):
    if random.random() < epsilon:
        return random.randrange(ACTIONS)
    else:
        return np.argmax(Q[state])
```

Fungsi ini digunakan untuk memilih aksi berdasarkan strategi ε-greedy: dengan probabilitas ε melakukan aksi acak (eksplorasi), sisanya memilih aksi terbaik dari Q-table (eksploitasi).
<br>

```
def train():
    epsilon = INITIAL_EPSILON
    game_state = GameState()

    for episode in range(EPISODES):
        # Reset game
        game_state = GameState()
        a_t = np.zeros(ACTIONS)
        a_t[0] = 1

        # Start game to get first state
        _, _, done = game_state.frame_step(a_t)

        state = get_features(game_state)
        total_reward = 0

        while not done:
            action = choose_action(state, epsilon)

            a_t = np.zeros(ACTIONS)
            a_t[action] = 1

            _, reward, done = game_state.frame_step(a_t)
            next_state = get_features(game_state)

            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            if done:
                td_target = reward
            else:
                td_target = reward + GAMMA * Q[next_state][best_next_action]

            td_error = td_target - Q[state][action]
            Q[state][action] += ALPHA * td_error

            state = next_state
            total_reward += reward

        print(f"Episode {episode} - Total reward: {total_reward} - Epsilon: {epsilon:.4f}")
```

**Penjelasan:**

- Mengulang game sebanyak EPISODES kali.
- Pada setiap episode, agen mengambil aksi berdasarkan Q-table atau eksplorasi.
- Q-table diperbarui menggunakan rumus Q-learning:
  $$
  Q(s, a) \leftarrow Q(s, a) + \alpha \left( r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right)
  $$
- Skor total per episode ditampilkan untuk melihat progres pelatihan.

## Referensi

Penugasan ini mengambil referensi dari repo berikut:

1. [yenchenlin/DeepLearningFlappyBird] (https://github.com/yenchenlin/DeepLearningFlappyBird)
