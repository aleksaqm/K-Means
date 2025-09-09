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
   
   1.1 Sekvencijalna verzija algoritma:
   - Početni centroidi se nasumično biraju iz skupa podataka.
   - Svaka tačka se dodeljuje najbližem centroidu.
   - Centroidi se ažuriraju računajući srednju vrednost svih tačaka koje im pripadaju.
   - Ponavljanje dok se ne postigne konvergencija ili maksimalan broj iteracija.
   
   1.2 Paralelizovana verzija algoritma:
   - Koristi `multiprocessing` biblioteku i deljenu memoriju (`shared_memory`) za skladištenje podataka.
   - Skup tačaka se deli na više blokova koji se obrađuju paralelno u različitim procesima.
   - Svaki proces računa udaljenosti i privremene rezultate za dodelu klastera i sumiranje tačaka po klasterima.
   - Glavni proces kombinuje rezultate i ažurira centre klastera.

2. **Rust implementacija**
   
   2.1 Sekvencijalna verzija algoritma:
   - Korišćenje `Vec<Point>` za skladištenje tačaka i centroida.
   - Dodela klastera i računanje novih centroida implementirano sa minimalnim memorijskim kopiranjem.
   - Praćenje maksimalnog pomeraja centroida radi konvergencije.
   
   2.2 Paralelizovana verzija algoritma:
   - Koristi se **Rayon** biblioteka (`par_iter_mut`) za jednostavnu i efikasnu paralelizaciju petlji.
   - Svaka tačka se paralelno obrađuje radi određivanja najbližeg centroida.
   - Sume i brojači za centroida se akumuliraju, a nakon završetka iteracije centroidi se ažuriraju.
   - Paralelizacija je „zero-cost“ jer Rayon automatski balansira opterećenje između niti i koristi radne stekove.

---

### Vizuelizacija
Jos treba razmisliti oko odabira biblioteke i nacina vizuelizacije...

### Očekivani rezultati
- Jasna implementacija sekvencijalne i paralelne verzije K-Means algoritma u Python-u i Rust-u.
- Upoređivanje performansi sekvencijalnog i paralelnog koda kroz eksperimente jakog i slabog skaliranja.
- Grafički prikazi ubrzanja u odnosu na broj procesorskih jezgara, sa idealnim (teorijskim) linijama skaliranja.
- Vizualizacija procesa klasterizacije kroz Rust aplikaciju.
- Kompletan izveštaj sa tehničkim detaljima, tabelama, grafikonima i analizom rezultata.




---

