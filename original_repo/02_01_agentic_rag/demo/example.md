User: Opowiedz mi o cache’u i gdzie został wspomniany.

Assistant:

### Co w tych materiałach oznacza „cache”

W notatkach AI_devs **cache** jest omawiany przede wszystkim jako mechanizm poprawy wydajności i obniżenia kosztów w systemach opartych na LLM-ach — szczególnie w agentach — którego celem jest **unikanie ponownego przetwarzania tych samych danych wejściowych**, a więc **zmniejszenie opóźnień i kosztu tokenów**.

Występują tu dwa ściśle powiązane znaczenia:

1. **Cache po stronie dostawcy/API (cache promptu / „prompt cache”)**
   - Dostawcy mogą buforować *niezmienione* części promptu pomiędzy zapytaniami, co obniża koszt i zauważalnie poprawia responsywność (krótszy **Time to First Token / TTFT**), o ile prefiks promptu pozostaje stabilny.  
   - Jest to traktowane jako optymalizacja **najwyższego priorytetu**.  
   Dowód: *S01E02* wprost wskazuje „Cache Promptu” jako priorytet i łączy go z TTFT oraz redukcją kosztów (S01E02, linie 236–246). Ostrzega też, że zmiana elementów dynamicznych (nawet data/czas) może natychmiast unieważnić cache (S01E02, linie 256–261).

2. **Cache po stronie aplikacji — buforowanie wyliczonych artefaktów/wyników**
   - Przykład: przy przetwarzaniu notatek/dokumentów, jeśli pliki pochodne zostały już wcześniej wygenerowane, aplikacja może pominąć ich ponowne generowanie — czyli „użyć cache”, wykorzystując zapisane wyniki.  
   Dowód: *S01E01* opisuje przykład, w którym wcześniej wygenerowane pliki oznaczają, że „nie ma potrzeby generować ich ponownie”, nazywając to „wykorzystaniem pamięci podręcznej” (S01E01, linie 309–315).

Kluczowym pojęciem operacyjnym jest **„cache hit”** — wysoki poziom ponownego użycia cache jest bezpośrednio powiązany z **szybkością i kosztem**, a decyzje projektowe (np. gdzie umieszczasz informacje dynamiczne) mają na to duży wpływ.  
Dowód: *S02E01* podkreśla „cache hit” jako priorytet wpływający na czas i koszt (S02E01, linie 160–169).

---

### Gdzie wspomniano cache (z lokalizacjami)

1. **S01E01 – jako kluczowa funkcja natywna dostawcy/API**
   - Cache jest wymieniany wśród „kluczowych natywnych funkcjonalności API” przy wyborze dostawcy; jego rola to ograniczanie ponownych obliczeń, kosztu i czasu odpowiedzi przy skali.  
   Źródło: `S01E01 - Programowanie interakcji z modelem językowym.md` (linie 189–196, szczególnie 194).

2. **S01E01 – przykład cache poprzez ponowne użycie wcześniej wygenerowanych plików**
   - Przykład „groundingu” pokazuje użycie cache: jeśli pliki wyjściowe już istnieją, nie generuj ich ponownie.  
   Źródło: `S01E01 ...` (linie 309–315).

3. **S01E02 – „prompt cache” jako najważniejsza optymalizacja szybkości/kosztu**
   - „Cache Promptu” jest opisany jako priorytet; automatyczny prompt caching (Gemini/OpenAI) obniża koszty i znacząco zmniejsza TTFT.  
   Źródło: `S01E02 - Techniki łączenia modelu z narzędziami.md` (linie 236–246).
   - Dodatkowa wskazówka praktyczna: dynamiczne elementy w system prompcie mogą psuć cache; lepiej utrzymywać treść systemową stabilną.  
   Źródło: `S01E02 ...` (linie 256–261 oraz sąsiednia dyskusja 256–274).

4. **S01E05 – cache jako fundament ekonomii i architektury produkcyjnej**
   - Wskazuje, że jedno zdarzenie użytkownika może uruchamiać wiele wywołań AI (nawet >1:50), więc cache staje się „fundamentalny”, choć nie zawsze możliwy.  
   Źródło: `S01E05 - Zarządzanie jawnymi oraz niejawnymi limitami modeli.md` (linie 89–94).
   - W przykładzie architektury agenta „prompt cache” jest wprost elementem logiki zarządzania kontekstem.  
   Źródło: `S01E05 ...` (linie 219–230).
   - Podkreśla, że utrzymanie poprawnego stanu agenta mocno wpływa na wykorzystanie prompt cache.  
   Źródło: `S01E05 ...` (linie 262–265).

5. **S02E01 – cache hit i jak struktura kontekstu może zniszczyć cache**
   - Ostrzega, że modyfikowanie system promptu w trakcie sesji to „prosty sposób” na problemy z prompt cache.  
   Źródło: `S02E01 - Zarządzanie kontekstem w konwersacji.md` (linie 34–39).
   - Zwraca uwagę, że narzędzia zwykle pojawiają się *pod* system promptem w oknie kontekstu — więc zmiana system promptu może wyrzucić definicje narzędzi z cache, pogarszając cache hit.  
   Źródło: `S02E01 ...` (linie 163–168).

---

### Przejrzane pliki
- `S01E01 - Programowanie interakcji z modelem językowym.md`
- `S01E02 - Techniki łączenia modelu z narzędziami.md`
- `S01E05 - Zarządzanie jawnymi oraz niejawnymi limitami modeli.md`
- `S02E01 - Zarządzanie kontekstem w konwersacji.md`

Jeśli chcesz, mogę wyciągnąć z tych fragmentów checklistę „reguł kciuka” (co trzymać statycznie, gdzie wkładać dane dynamiczne, jak utrzymać wysoki cache hit).