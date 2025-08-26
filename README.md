# Projekat iz Naprednih Tehnika Programiranja

## Ocena za koju se radi projekat: 10

---

## Tema: K-Means algoritam i njegova paralelizacija

### Opis
K-Means je jedan od najpoznatijih algoritama nenadgledanog učenja, koji se koristi za grupisanje podataka u unapred definisan broj klastera \(K\). Algoritam funkcioniše iterativno:
1. Nasumično bira početne centre klastera (tzv. centroidi).
2. Svakoj tački dodeljuje klaster na osnovu minimalne euklidske udaljenosti od centroida.
3. Računa nove centre klastera kao sredinu svih tačaka koje im pripadaju.
4. Ponavlja korake 2 i 3 dok se centri ne stabilizuju ili dok se ne dostigne unapred definisan broj iteracija.

K-Means se široko koristi u analizi podataka, obradi slike, mašinskom učenju i različitim aplikacijama gde je potrebno pronaći obrasce u velikim skupovima podataka. Algoritam obrađuje veliki broj tačaka i pogodan je za paralelizaciju.

---

### Verzije algoritma

1. **Python implementacija**
   
   1.1 Sekvencijalna verzija algoritma.
   
   1.2 Paralelizovana verzija algoritma korišćenjem `multiprocessing` biblioteke.

2. **Rust implementacija**
   
   2.1 Sekvencijalna verzija algoritma u programskom jeziku Rust.
   
   2.2 Paralelizovana verzija algoritma korišćenjem niti
---

### Očekivani rezultati
- Jasna implementacija sekvencijalne i paralelne verzije K-Means algoritma u Python-u i Rust-u.
- Upoređivanje performansi sekvencijalnog i paralelnog koda kroz eksperimente jakog i slabog skaliranja.
- Grafički prikazi ubrzanja u odnosu na broj procesorskih jezgara, sa idealnim (teorijskim) linijama skaliranja.
- Vizualizacija procesa klasterizacije kroz Rust aplikaciju.
- Kompletan izveštaj sa tehničkim detaljima, tabelama, grafikonima i analizom rezultata.

---

