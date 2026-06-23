# Game Review BERTopic Topic 0 Refinement

- Generated: 2026-06-23T19:09:01
- Input: `C:\Users\admin\Documents\studable query latent\game_review_data\game_review_cleaned_3_sentences`
- Base sample documents: 100000
- Parent topic selected: 0
- Parent Topic 0 documents: 96691
- Parent outlier documents: 2884
- Embedding dimension: 1024

## Parent Model

- UMAP `n_neighbors`: 100
- HDBSCAN `min_cluster_size`: 100
- HDBSCAN `min_samples`: default
- HDBSCAN `cluster_selection_method`: `eom`
- CountVectorizer `min_df`: 1

## Subtopic Model

- Documents: 96691
- UMAP `n_neighbors`: 30
- HDBSCAN `min_cluster_size`: 50
- HDBSCAN `min_samples`: 10
- HDBSCAN `cluster_selection_method`: `leaf`
- CountVectorizer `min_df`: 5
- Random state: 42
- Fit time: 141.4 seconds

Note: these topic IDs are second-level IDs inside parent Topic 0. They are not the same namespace as the parent model topic IDs.

## Environment

| Package | Version |
|---|---:|
| `bertopic` | `0.17.4` |
| `hdbscan` | `0.8.44` |
| `umap-learn` | `0.5.12` |
| `ijson` | `3.5.0` |
| `numpy` | `2.4.3` |
| `scikit-learn` | `1.8.0` |

## Summary

- Subtopics excluding outliers: 206
- Outlier documents: 67523
- Outlier rate: 69.83%

## Subtopic Table

| Subtopic | Count | Top Words |
|---:|---:|---|
| -1 | 67523 | outliers |
| 0 | 939 | music, soundtrack, sound, audio, sound design, sounds, tracks, ost, design, song |
| 1 | 704 | hours, stunden, horas, hours game, game hours, took, hour, ive, ich, finish |
| 2 | 630 | sale, price, worth, preo, buy, wait, precio, worth price, worth money, cheaper |
| 3 | 590 | price, sale, worth, game worth, buy, preo, recommend, bought, buy game, buying |
| 4 | 563 | combat, fights, combate, fighting, fight, game combat, fun, responsive, battles, satisfying |
| 5 | 503 | enemies, attacks, dodge, enemy, attack, parry, damage, gegner, melee, dodging |
| 6 | 480 | map, maps, mapa, mapas, minimap, markers, die, map design, location, world |
| 7 | 457 | skill, skills, tree, level, crafting, gear, perks, upgrade, xp, points |
| 8 | 438 | cd, dps, boss, aoe, shadow, fromsoftware, aiai, debuff, fs, p5 |
| 9 | 412 | thats, say, lets, dont know, thing, let, know, important, im, mean |
| 10 | 408 | early access, access, early, release, beta, released, developers, alpha, game, content |
| 11 | 386 | cars, car, racing, wheels, hot, driving, drive, vehicle, tracks, vehicles |
| 12 | 383 | review, reviews, comments, negative, write, comment, read, positive, feel free, ill |
| 13 | 363 | controller, mouse, keyboard, controls, button, press, control, buttons, xbox, keys |
| 14 | 346 | animals, hunting, deer, animal, hunter, dog, pet, cotw, die, eat |
| 15 | 333 | boss, bosses, fight, boss fights, fights, final boss, final, patterns, beat, challenging |
| 16 | 327 | story, plot, ending, good story, predictable, interesting, written, stories, writing, good |
| 17 | 319 | npc, ff, gg, arrive, ta, cast, doom, xp, powerful, max |
| 18 | 305 | steam, origin, launcher, ea, files, install, launch, download, delete, game steam |
| 19 | 300 | quests, quest, tiles, story quests, tile, main, main story, story, complete, haunting |
| 20 | 285 | minutos, una, hasta, luego, la, las, durante, para, meio, el |
| 21 | 284 | fun, good game, game fun, amazing, fun play, great game, game game, amazing game, great, game |
| 22 | 276 | sacrifice, ta, deserves, rich, weight, bad, poor, hand, life, end |
| 23 | 269 | missions, mission, story missions, objective, main, misiones, story, doing, missionen, objectives |
| 24 | 261 | rogue, rpg, roguelite, roguelike, warhammer, legacy, marauders, elements, escape, similar |
| 25 | 261 | deck, cards, card, karten, energy, sts, qe, turn, round, draw |
| 26 | 260 | dlc, jrpg, lessons, katana, time got, fortune, rpg, deaths, qte, aoe |
| 27 | 258 | standalone, goh, returnal, ps2, sony, suffering, uncharted, sekiro, hearts, dps |
| 28 | 254 | ps |
| 29 | 239 | ai, ki, dumb, ia, bots, bad, stupid, just bad, improved, die |
| 30 | 238 | devs, developers, update, updates, entwickler, feedback, developer, dev, die entwickler, community |
| 31 | 236 | recommend game, review, recommend, reviews, negative, negative review, review game, current state, game, state |
| 32 | 236 | dlc, dlcs, content, upcoming, prices, base game, paid, released, price, final |
| 33 | 227 | jeu, le, et, le jeu, pas, je, pour, ne, qui, les |
| 34 | 222 | voice, voice acting, acting, voices, lines, english, dialogue, characters, great, job |
| 35 | 215 | english, translation, language, russian, translate, google, las, german, oyun, auf |
| 36 | 214 | characters, personality, character, memorable, cast, charaktere, character development, written, depth, unique |
| 37 | 210 | ban, hackers, banned, discord, cheating, account, post, forums, reddit, community |
| 38 | 209 | flaws, bad game, game isnt, bad, design, game, mechanics, missing, gameplay, good game |
| 39 | 208 | ai, amidst, echo, world, artificial, num, essence, shape, words, thank |
| 40 | 202 | roguelike, boss, dream, t1, sr, code, xd, alpha, tm, buff |
| 41 | 201 | love game, love, like game, game, enjoyed, loved, game love, game really, happy, knew |
| 42 | 195 | price, average, waste money, sua, kids, waste, short, long, infinito, replay |
| 43 | 192 | servers, server, connection, cross, ping, platform, issues, play, online, los |
| 44 | 192 | boss, npc, deus, ex, qte, npc npc, buff, ai, combo, ui |
| 45 | 181 | ea, franchise, electronic, arts, company, launch, post, year, killing, player base |
| 46 | 181 | remaster, remake, remastered, remnant, original, collection, kingdoms, job, improves, original game |
| 47 | 176 | souls, soulslike, dark souls, dark, mortal, shell, hollow, knight, fromsoftware, sekiro |
| 48 | 169 | ships, ship, salvage, boat, pulling, shifts, apart, cut, right, enemy |
| 49 | 168 | french, st, key, el, en el, maestra, poor, revolution, lejos, oficial |
| 50 | 163 | bug, mod, gmod, online, auf, bedeutet, gefhl, zwei, unreal, aus |
| 51 | 148 | boss, rush, il |
| 52 | 148 | raw, ore, npc |
| 53 | 146 | jump scare, scare, boss, jump, dlc |
| 54 | 144 | nasa, computer, ask, spare, press, difficulty, easy, short hours, size, easy learn |
| 55 | 144 | graphics, grafik, die grafik, grficos, os, beautiful, graphically, personagens, qe, good |
| 56 | 143 | bug, ui, west, mod, npc, fps, ai, dlc |
| 57 | 141 | chinese, clans, darktide, hard work, npc, listed, google, translation, tm, board |
| 58 | 139 | recommend, recommended, recommendation, highly, caveats, highly recommend, definitely recommend, recomendado, recomendo, yeah |
| 59 | 137 | save, reload, load, saving, saves, file, manual, manually, quit, save file |
| 60 | 137 | stardew, valley, crossing, animal, farming, haven, sun, storm, sim, harvest |
| 61 | 135 | matchmaking, lobby, match, lobbies, matches, players, queue, spieler, amigos, bots |
| 62 | 134 | town, city, environment, beautiful, gorgeous, environments, scenery, look, lake, towns |
| 63 | 134 | crops, farming, ranch, farm, plant, simulation, harvest, cheese, fields, grow |
| 64 | 133 | pvp, pve, mode, server, ayr, team, players, solo, porque, rankings |
| 65 | 132 | ui, menu, interface, main menu, menus, confusing, clunky, horrible, main, buttons |
| 66 | 132 | p5, persona, jrpgs, ps5, turnbased, original, given, hard place, story content, japan |
| 67 | 131 | building, base, buildings, build, houses, wood, bauen, man, market, house |
| 68 | 131 | dark souls, dark, souls, grind, difficult, quase, difcil, bueno, tu, historia |
| 69 | 130 | disappointing, disappointment, shame, frustrating, pathetic, annoying, unfair, annoyance, sad, atrocious |
| 70 | 129 | ga, bethesda, warning, ps, demo, fall, bug, pc, se, game |
| 71 | 122 | spieler, tekken, ich, von, der, habe, wissen, nachdem, um die, von der |
| 72 | 120 | elden ring, elden, chill, ring, man |
| 73 | 119 | fantasy, ff, remake, jrpg, final, play original, rpg, playstation, launched, game thats |
| 74 | 118 | gold, currency, gems, bars, bounty, buy, role, roles, daily, ingame |
| 75 | 117 | dragon, series, remakes, ps2, entries, remastered, jrpg, la, franchise, games |
| 76 | 117 | dlc, plus, horror, original, horror game, visual, novel, timings, slightly different, summarize |
| 77 | 116 | npc, npc npc, motive, jump scare, scare, jump, city |
| 78 | 114 | classes, class, classe, mage, playstyle, character, different, choose, differently, types |
| 79 | 112 | star, wars, stars, sea, ii, fan, flight, fans, qe, universe |
| 80 | 111 | paint, check, decent, fast, run, consegue, seu, meh, da pra, ms |
| 81 | 110 | horror, horror game, cyberpunk, scary, atmosphere, psychological, spooky, movies, scare, jump |
| 82 | 110 | fps, settings, frame, drop, drops, frames, max, highest, runs, high |
| 83 | 110 | guns, rifle, gun, shotgun, ammo, pistol, reload, clean, weapon, laser |
| 84 | 108 | coop, play, online, multiplayer, works, pvp, friends, mode, communication, matchmaking |
| 85 | 107 | fishing, fish, catch, minigame, click, boat, catching, seas, simple, creatures |
| 86 | 104 | et, vous, des, les, le, armes, une, qui, pour, pas |
| 87 | 102 | easy learn, master, learn, easy, hard, dominar, aprender, fcil, difcil, leicht |
| 88 | 102 | weather, rain, lights, season, lighting, park, storm, winds, night, light |
| 89 | 101 | multiplayer, modo, online, opinio, traffic, coop, der, mode, gleiche, haha |
| 90 | 100 | gears, tactics, tiene, games like, veteran, foes, war, juego, pero, lo que |
| 91 | 100 | vr, wanted, campaign, tech, virtual, modding, set pieces, wanted like, controls, sandbox |
| 92 | 100 | puzzles, puzzle, solve, solving, freedom, logical, figure, relaxing, feeling, felt |
| 93 | 100 | npc, tm, potion, ps, sad, craft |
| 94 | 99 | recommend, recommend game, game recommend, highly, highly recommend, game highly, definitely recommend, recommendation, wholeheartedly, recommended |
| 95 | 99 | bad, good, bad thing, execution, perfect, good stuff, good bad, para lo, good great, forma |
| 96 | 97 | open world, simulator, open, world, openworld, sim, walking, simulation, sims, ak |
| 97 | 96 | mod, easy |
| 98 | 95 | difficulty, hard, normal, hardest, normal difficulty, dificuldade, easy, difficult, challenge, settings |
| 99 | 95 | mount, zero, npc |
| 100 | 94 | traverse, aiai, ps, pro, pvp |
| 101 | 93 | goh, ga, evil, studio, update |
| 102 | 93 | god war, favorite, favourite, god, game year, best, games, favorite games, played, year |
| 103 | 91 | cat, robots, hunger, companion, surface, scratch, tries, cyberpunk, bar, stuff |
| 104 | 90 | ps4, pc, xbox, ps5, playstation, hoje, aquele, console, ps3, um |
| 105 | 90 | camera, view, zoom, cockpit, position, movement, person, mouse, adjust, controls |
| 106 | 90 | visually, visuals, graphics, stunning, art, visual, gorgeous, style, beautiful, art style |
| 107 | 90 | hidden, secret, exploration, loot, secrets, easter eggs, easter, rewarded, eggs, explore |
| 108 | 89 | funny, laugh, amusing, actually pretty, jokes, humor, joke, hilarious, loud, pretty |
| 109 | 89 | horas, short hours, hours long, hours, entre, hours average, long, largo, curto, hours short |
| 110 | 89 | come close, satisfactory, aiai, program, planet, ps, goal, fit, end game, wouldnt |
| 111 | 89 | runescape, account, old school, school, evolution, accounts, old, mmos, content, updates |
| 112 | 88 | ordinary, jump scare, gmod, scare, universal, dawn, stopped, energy, jump, server |
| 113 | 88 | tm |
| 114 | 88 | thank, thanks, reading, gracias, awards, read, haha, spending time, vielen, por |
| 115 | 86 | alien, crusade, named, hero, planet, humanity, king, humans, demons, galaxy |
| 116 | 86 | broadcast, news, angles, charge, right time, japan, cameras, room, influenced, ensuring |
| 117 | 86 | patch, fixed, fix, patches, issues, hopefully, notes, june, fixes, fix issues |
| 118 | 85 | heit, wissen, granted, dass die, auf dem, breath, suddenly, alle, schon, dass |
| 119 | 84 | precisa, pra, jogar, jogabilidade, dois, pra jogar, mnimo, junto, pra zerar, zerar |
| 120 | 84 | good, neat, phenomenal, looks good, diese, just good, alles, einfach nur, einfach, good job |
| 121 | 84 | dinheiro, promoo, espere, uma promoo, se tiver, dinheiro sobrando, sobrando, compre, tiver, fcil |
| 122 | 82 | achievements, achievement, completion, tied, online, features, completing, steam, trial, wait |
| 123 | 82 | ban, robot, gua, dos, want spend, dare, spend money, welcome, dont want, money |
| 124 | 82 | healing, health, heal, hp, bar, ki, charge, automatically, perform, damage |
| 125 | 79 | animations, models, animation, character, sh, movements, vibrant, style, detailed, ive seen |
| 126 | 79 | rpg |
| 127 | 78 | pizza, ga, ni, crew, produced, featuring, live, create, yo, box |
| 128 | 78 | enjoyed, liked, loved, enjoy, love, thoroughly enjoyed, personally, absolutely, surprised, thoroughly |
| 129 | 78 | ate, open, learned, puzzle, beginning, save, nem, mit, die |
| 130 | 77 | cook, aoe |
| 131 | 77 | mi |
| 132 | 75 | fs, jrpg, boss, hunter, rpg |
| 133 | 73 | monster hunter, hunter, monster, rise, rises, world, sunbreak, stories, srie, monsters |
| 134 | 73 | furniture, lots, decoration, room, house, table, takes long, people getting, decorations, want |
| 135 | 73 | histria, uma, tem uma, tem, dizer que, dizer, sobre, bem, vida, que tem |
| 136 | 72 | walking, walk, speed, head, day night, dry, sua, night, game runs, sleep |
| 137 | 72 | art, art direction, visuals, direction, art style, stunning, artstyle, style, piece, absolutely stunning |
| 138 | 70 | f1, cars, really love, vas, improvements, classic, engine, ksmlar, eu, vou |
| 139 | 70 | accuracy, cqb, rainbow, dps, damage |
| 140 | 70 | microtransactions, pay win, pay, win, dollars, ter, limited time, deals, ter um, contracts |
| 141 | 70 | ping, bug, red, tu, cantidad, steam, segn, ps, question, hasta |
| 142 | 69 | story lore, youll need, average good, story story, grinding, second, lore, average, lovely, story |
| 143 | 69 | mods, modding, mod, modders, community, mod support, custom, active, ac, plenty |
| 144 | 68 | fps, settings, performance, high, running, stable, patch, quando, usage, medium |
| 145 | 68 | flight, flying, fly, planes, plane, simulator, real, real world, real life, simulation |
| 146 | 66 | et, pas, les, dans, qui, peu, une, le, cest, des |
| 147 | 66 | necessary progress, isnt necessary, necessary, progress, isnt, protocol, callisto, mandatory, arc, roadmap |
| 148 | 65 | gta, hk, gmod, minecraft, ea, bug, dice, heart, pro, cut |
| 149 | 65 | deluxe, pack, trilogy, assassination, edition, access, campaigns, locations, sins, goty |
| 150 | 65 | jump scare, scare, jump, qte, npc |
| 151 | 65 | customization, character, appearance, hair, options, male, color, female, allows, customisation |
| 152 | 65 | weapons, weapon, armas, different, guns, types, waffen, prefer, man eine, selection |
| 153 | 65 | climb, zone, climbing, oft, luft, walls, rock, jump, dead, fall |
| 154 | 64 | rpg, pc |
| 155 | 63 | mafia, remake, ii, ich, war, hat, das, edition, bin, von |
| 156 | 63 | original game, original, improved, titles, better, previous titles, game better, better game, previous, jogo |
| 157 | 63 | npcs, npc, talk, npc npc, talks, fleshed, relationship, friendship, character game, characters |
| 158 | 63 | payday, heist, eu, just play, lanamento, que eu, launch, da franquia, reais, downgrade |
| 159 | 63 | slime, science, new, ones, returning, continue, added, exploration, new ones, raise |
| 160 | 62 | factions, faction, jede, different, lasst, sich, society, aus, die, mit |
| 161 | 62 | friends, solo, play friends, play, game play, want play, playing, game best, youre playing, ive lot |
| 162 | 62 | dead space, dead, space, remake, visceral, horror, original, space game, fans, fan |
| 163 | 61 | father, mother, sou, anos, affair, agora, main character, meu, tragic, victim |
| 164 | 60 | dificuldade, difficulty, normal, voc, easy, slider, hard, good great, difficulties, zor |
| 165 | 60 | resources, resource, management, recursos, receive, setzt, man, materials, demand, sources |
| 166 | 60 | ui, alt, npc |
| 167 | 60 | mod, mob, steam, einfacher, runter, restored, ps1, buddy, sign, ich mir |
| 168 | 60 | hey, machine, ask, attack |
| 169 | 60 | lore, russian, story, rights, presented, narrative, storylines, tremendous, interesting, packed |
| 170 | 60 | cutscenes, cutscene, scenes, cut scenes, cut, skip, scene, dialogue, skipping, sono |
| 171 | 60 | persona, royal, golden, pc, played, na, na sua, jrpgs, minha, vida |
| 172 | 59 | ip, boss |
| 173 | 59 | no valid tokens |
| 174 | 59 | level design, design, level, levels, designed, decoration, verticality, sont, puzzle, shortcuts |
| 175 | 59 | worth price, price free, free, price, worth, gratis, lo que, vale la, el precio, la pena |
| 176 | 59 | dune, sagit, rpg, period, medieval, il, le monde, tout le, qualit, qui |
| 177 | 59 | king, kingdom, heaven, realm, sons, son, holy, led, sent, family |
| 178 | 58 | victory, plan, hero, ground, comes, place, free |
| 179 | 58 | city, storm, direction, jack, research, passing, calling, successor, builder, cold |
| 180 | 58 | replay, replay value, replayability, value, tutorial, tutorials, endless, jest, replayable, und man |
| 181 | 58 | players, playerbase, player base, new players, new player, player experience, player, new, arrived, base |
| 182 | 58 | optimization, performance, performance issues, optimisation, issues, kt, quedas, desempenho, pcs, frequent |
| 183 | 57 | dos |
| 184 | 57 | left recommended, money left, spare money, recommended, spare, left, money, spend, cautious, spend money |
| 185 | 57 | grind, grinding, average, level, grindy, replayability, requires, game time, stone, hour |
| 186 | 57 | crashes, bugs, crash, encountered, gamebreaking, bei mir, experienced, didnt, glitches, havent |
| 187 | 57 | campaign, campanha, campaigns, gesagt, gut, achievement, tell, wie, auch, hard place |
| 188 | 57 | oil, steering, tie, missing, belt, engine, remove, rack, rust, tire |
| 189 | 56 | true |
| 190 | 56 | cup, replace, long, average, itll, game time, short, price, life, time |
| 191 | 55 | pass, battle, pase, battle passes, ac, konnte, passe, passes, se, eyler |
| 192 | 55 | cup, replace, itll, game time, long, life, time, para uma, short, suficiente para |
| 193 | 55 | cosmetics, cosmetic, shop, items, ingame, expensive, money, store, real, avatar |
| 194 | 55 | htte, wish, look forward, forward, future, grows, projects, alles, ich bin, man |
| 195 | 54 | mode, modes, ranked, training, game modes, practice, frame, data, single player, gamemodes |
| 196 | 54 | improvement, improved, improvements, content, think, improving, finished, unfinished, puede, gibt es |
| 197 | 54 | village, island, villagers, towns, expand, town, visit, resources, build, building |
| 198 | 54 | aliens, alien, marines, fan, movie, universe, descent, elite, shooters, coop |
| 199 | 53 | hades, ii, curse, rating, picking, repeat, bem, runs, meta, expecting |
| 200 | 53 | resident, silent, evil, hill, souls, horror, survival horror, survival, alten, puzzles |
| 201 | 52 | choices, affect, story, impact, choice, matter, narrative, decisions, regardless, sooner |
| 202 | 51 | sequel, worthy, series, suite, successor, looking forward, hope, forward, seasons, storyline |
| 203 | 50 | bugs, itens, game breaking, bug, breaking, juego en, jogo, simplesmente, suelo, breaking bugs |
| 204 | 50 | films, est, pas, qui, ce, pour, aussi, le, clairement, qui ne |
| 205 | 50 | tekken, heat, melhor que, opponent, aggressive, identity, nod, dio, anteriores, que |

## Top Subtopic Examples

### Subtopic 0 (939 docs)

Top words: music, soundtrack, sound, audio, sound design, sounds, tracks, ost, design, song

- `1012790_10355.json review=13 sentence=sentence_15`: To add to that stomach churning sea of what the f'k, you then add the sound design.
- `1012790_10355.json review=13 sentence=sentence_18`: It's very ambient...there's a befouled wind blowing...chairs making creepy classroom noises...something unnatural either creaking, or sucking out a soul, hard to tell...etc. etc.
- `1012790_10355.json review=13 sentence=sentence_19`: Kudos to the sound design...you're all sick puppies.

### Subtopic 1 (704 docs)

Top words: hours, stunden, horas, hours game, game hours, took, hour, ive, ich, finish

- `1009290_6672.json review=4 sentence=sentence_5`: Se você não rushar a historia, você leva no minimo 100 horas pra zerar o jogo igual a mim ;D
- `1009290_6672.json review=5 sentence=sentence_3`: I am halfway through the main campaign with ~50 hours of play-time which is well above average for campaigns nowadays.
- `1009290_6672.json review=5 sentence=sentence_7`: There's actually a lot of content that you could play Co-op, but if you do a 'long-play' through Chapter 1 (not skipping any dialogue and choosing to opt for the 'long-play' option) then expect anywhere from 10-18 hours for Chapter 1 to be completed.

### Subtopic 2 (630 docs)

Top words: sale, price, worth, preo, buy, wait, precio, worth price, worth money, cheaper

- `1012790_10355.json review=15 sentence=sentence_2`: the devs are offering this on the cheap.
- `1016950_4196.json review=3 sentence=sentence_54`: Normally I’d be alright with this but these prices are ridiculous.
- `1016950_4196.json review=4 sentence=sentence_7`: Oh, but you’re here to know if it is worth the money and time investment?

### Subtopic 3 (590 docs)

Top words: price, sale, worth, game worth, buy, preo, recommend, bought, buy game, buying

- `1012790_10355.json review=5 sentence=sentence_3`: I bought this game last night.
- `1012790_10355.json review=11 sentence=sentence_13`: BEFORE YOU BUY THIS GAME -- research it.
- `1016800_13368.json review=3 sentence=sentence_15`: Either way, this is a great game on its own, especially for the price.

### Subtopic 4 (563 docs)

Top words: combat, fights, combate, fighting, fight, game combat, fun, responsive, battles, satisfying

- `1016800_13368.json review=7 sentence=sentence_48`: 敵兵のアルゴリズムも複雑ではないので、後半になるとコソコソするのが面倒になってしまう。
- `1029690_19693.json review=7 sentence=sentence_7`: Authentic.-I NEVER engage in combat and play every mission undetected.
- `1030210_11120.json review=8 sentence=sentence_4`: The combat is smooth apart from just clunky movement at times.

### Subtopic 5 (503 docs)

Top words: enemies, attacks, dodge, enemy, attack, parry, damage, gegner, melee, dodging

- `1009290_6672.json review=5 sentence=sentence_31`: You get to block incoming attacks, buff allies, etc.
- `1012790_10355.json review=3 sentence=sentence_14`: You almost fight exclusively like 2 enemies ever, which is a shame cause there's a whole lot of them to use and fighting against some of them just becomes a hassle in environments that make fighting either way too close or way too far (staying vague so I don't spoil enemy discovery cause that's a lot of the fun).
- `1012790_10355.json review=6 sentence=sentence_17`: I quickly moved back outside, where a collection of monsters that was chasing me across the map has gathered.

### Subtopic 6 (480 docs)

Top words: map, maps, mapa, mapas, minimap, markers, die, map design, location, world

- `1012790_10355.json review=3 sentence=sentence_18`: As a student in game design myself, I can attest that is NO small task to do, throwing out months of work building, testing, and refining an ENTIRE WORLD MAP simply to make it better.
- `1012790_10355.json review=3 sentence=sentence_20`: Now the world map is split into sections, all of which have multiple entry points (already an improvement), interconnect to each other, and some of which connect directly to your base.
- `1012790_10355.json review=3 sentence=sentence_23`: And even more than that, maps can change according to you mission to add even more, different varieties to you travels.

### Subtopic 7 (457 docs)

Top words: skill, skills, tree, level, crafting, gear, perks, upgrade, xp, points

- `1009290_6672.json review=5 sentence=sentence_15`: I think when you fully unlock the magic system you get access to 24 new abilities essentially.
- `1009290_6672.json review=5 sentence=sentence_19`: Like yeah, there is a ring from the premium store that can give you a chance for extra drops or more experience earned, but those mechanics also exist in the game and if you really want to boost your drops, just turn the difficulty up since drops scale off of difficulty...
- `1009290_6672.json review=5 sentence=sentence_34`: So you have to level up which is easier to do in SAO and enjoy than X-Men Legends.---------

### Subtopic 8 (438 docs)

Top words: cd, dps, boss, aoe, shadow, fromsoftware, aiai, debuff, fs, p5

- `1009290_6672.json review=3 sentence=sentence_13`: 第一，把装备全部换了，主力去提高命中，放弃其他属性。
- `1009290_6672.json review=3 sentence=sentence_29`: 简单来说，我自己一个人一套一千万，万一中间伙伴放一个剑技，我就剩下五百万伤害。
- `1009290_6672.json review=3 sentence=sentence_36`: 后期大家属性都那么高了，怪打起来都MISS好伐！

### Subtopic 9 (412 docs)

Top words: thats, say, lets, dont know, thing, let, know, important, im, mean

- `1009290_6672.json review=5 sentence=sentence_50`: It is A LOT... way too much to cover here though.
- `1009290_6672.json review=7 sentence=sentence_42`: 이제 뭐가 문제인지 아실 거 같나요?
- `1012790_10355.json review=13 sentence=sentence_41`: That's what I tell myself at least...

### Subtopic 10 (408 docs)

Top words: early access, access, early, release, beta, released, developers, alpha, game, content

- `1012790_10355.json review=8 sentence=sentence_6`: 楽しめたならbeta branchから1.0もプレイして欲しいです。
- `1016800_13368.json review=9 sentence=sentence_4`: Bei Chernobylite kann man sehr gut sehen, wie EarlyAcces gut und informativ funktionieren kann - es gibt wöchentliche Updates der Entwickler und auch auf dem Youtube-Channel kann man den Fortschritt sehr gut mit verfolgen.
- `1016950_4196.json review=9 sentence=sentence_5`: At release it was in early, early beta phase at best (and that's a very generous description).

### Subtopic 11 (386 docs)

Top words: cars, car, racing, wheels, hot, driving, drive, vehicle, tracks, vehicles

- `1029690_19693.json review=9 sentence=sentence_4`: Arabanın içinde bir npc var araca sıkıyorsun öylece içinde oturuyor.
- `1030830_26263.json review=5 sentence=sentence_10`: lcw den cekette alınabilir-80'li prima çocuk bezi alınıp bir baba mutlu edilebilir-2.5 kilo lüks karışık kuruyemiş alınabilir-3-4 kilo levrek alınıp dostlarla rakı masasında buluşulabilir-arabanızın motor yağı ve hatta yağ filtresi değiştirilebilir-12 paket sigara alınabilir
- `1030840_80583.json review=7 sentence=sentence_12`: Оружие не чувствуется, автомобили тоже, вроде бы физика и есть, но врезаются они кринжова...

### Subtopic 12 (383 docs)

Top words: review, reviews, comments, negative, write, comment, read, positive, feel free, ill

- `1009290_6672.json review=7 sentence=sentence_24`: 자세한 건 나중에 리뷰를 수정해 추가적으로 쓰겠습니다.3.
- `1016800_13368.json review=3 sentence=sentence_3`: (Which might be the reason for the negative reviews.
- `1016950_4196.json review=7 sentence=sentence_21`: They need to earn our trust back we MUST wait for reviews.

### Subtopic 13 (363 docs)

Top words: controller, mouse, keyboard, controls, button, press, control, buttons, xbox, keys

- `1009290_6672.json review=8 sentence=sentence_38`: Steam側でのコントローラー設定を強制ONにしないと、Steam側でのコントローラーサポート機能が動作しない。
- `1012790_10355.json review=4 sentence=sentence_18`: SIMPLE SETTING WOULD-oh wait i already did my rant...anyway.
- `1029780_17642.json review=9 sentence=sentence_21`: ich platze vor Genervt-sein, keine Blueprintfunktion, Tastenbelegung ist rudimetär)- Hitzewellen, Kältewellen, Thors Donner etc sind NICHT anwählbar.

### Subtopic 14 (346 docs)

Top words: animals, hunting, deer, animal, hunter, dog, pet, cotw, die, eat

- `1016800_13368.json review=6 sentence=sentence_13`: the catch is you have to manage them, starting with things like making sure youve scavenged enough for them to eat, and youve provided nice enough rest facilities.
- `1016800_13368.json review=10 sentence=sentence_28`: Ihr sorgt dafür, dass sie genug Essen haben und einen Platz zum schlafen.
- `1029780_17642.json review=6 sentence=sentence_40`: Супер урожай осенью, плодовитость весной (больше зайцев, оленей (можно кстати добавить больше живности)).

### Subtopic 15 (333 docs)

Top words: boss, bosses, fight, boss fights, fights, final boss, final, patterns, beat, challenging

- `1030210_11120.json review=9 sentence=sentence_16`: Don't bother doing these, as they bug out the majority of the time and the boss doesn't spawn.
- `1036890_4908.json review=11 sentence=sentence_13`: Actually has some of the better bosses in the series, but there's only 2 this time.
- `1049410_23830.json review=5 sentence=sentence_36`: There are only a handful of puzzles I didn't enjoy very much, and they're in what is intentionally considered as "the boss" levels.

### Subtopic 16 (327 docs)

Top words: story, plot, ending, good story, predictable, interesting, written, stories, writing, good

- `1016800_13368.json review=7 sentence=sentence_57`: 個々の要素はコンパクトだがストーリーを軸に手堅くまとまっており、完成度が高い。
- `1016800_13368.json review=9 sentence=sentence_11`: Die Story von Igor und Tatyana wird spannen umgesetzt und fordert einen stehts weiter zu machen um ihr wieder näher zu kommen->
- `1016800_13368.json review=12 sentence=sentence_5`: L'histoire est prenante et certains plans magnifiques ( mention honorable pour le mode photo)

### Subtopic 17 (319 docs)

Top words: npc, ff, gg, arrive, ta, cast, doom, xp, powerful, max

- `1009290_6672.json review=3 sentence=sentence_132`: 没看过的做好心理准备会有多出戏，看过的可以通过剧情去回顾。
- `1009290_6672.json review=3 sentence=sentence_195`: 原创女主的存在意义，完全就是弄一堆无趣沉闷的原创剧情来消耗你的时间。
- `1009290_6672.json review=3 sentence=sentence_196`: 我所有剧情一字一句的看没跳过，最后发现自己是个撒比。

### Subtopic 18 (305 docs)

Top words: steam, origin, launcher, ea, files, install, launch, download, delete, game steam

- `1009290_6672.json review=8 sentence=sentence_16`: インストール先にある「sao_al.exe」のプロパティを開き、「管理者として実行」にマーク。
- `1030840_80583.json review=14 sentence=sentence_2`: 解决方法如下（简体中文设置方式）：①在游戏本地文件夹内启动“mafiadefinitiveedition.exe”；②点开“选项”③选择“游戏”，然后列表往下拉，看到“禁用启动器”；④“禁用启动器”设置为“打开”；⑤就可以退出设置了，下次进入游戏时，就可以直接从Steam上启动游戏了。
- `1034140_19644.json review=4 sentence=sentence_14`: Not right now unless you you really like hentai games and just want to support them on steam so they don't get censored/banned.

### Subtopic 19 (300 docs)

Top words: quests, quest, tiles, story quests, tile, main, main story, story, complete, haunting

- `1009290_6672.json review=5 sentence=sentence_35`: So in Skyrim, you get a main quest to go clear out some bandits and the only other time you interact with the quest-giver is when you finished the quest... and you're like:
- `1016800_13368.json review=11 sentence=sentence_18`: Alle bisherigen Erfolge (!) in der Tagesquest beim Kampf, gefundene Dinge, Loot usw. müssen neu erarbeitet und gefunden werden.
- `1030210_11120.json review=9 sentence=sentence_7`: You pick up a quest from an NPC with little-to-no explanation as to why you are doing it in the first place, go to a location, kill a monster, teleport back to the NPC, who will then give you another quest to go to a boring, copy-and-paste dungeon to kill another monster boss.

### Subtopic 20 (285 docs)

Top words: minutos, una, hasta, luego, la, las, durante, para, meio, el

- `1041720_6700.json review=14 sentence=sentence_50`: Maybe heat up a can of refried beans or something".
- `1057090_26450.json review=12 sentence=sentence_11`: You can run it on a microwave✅ Average✅ High end
- `1110910_8180.json review=3 sentence=sentence_12`: Apoya el microondas en la heladera

### Subtopic 21 (284 docs)

Top words: fun, good game, game fun, amazing, fun play, great game, game game, amazing game, great, game

- `1012790_10355.json review=8 sentence=sentence_32`: 浮いたオブジェクトなど目立ちますが、総合的には良ゲーです。
- `1012790_10355.json review=11 sentence=sentence_2`: 10/10, Wonderful Game
- `1012790_10355.json review=13 sentence=sentence_32`: And that's a good thing in this type of game...trust me.

### Subtopic 22 (276 docs)

Top words: sacrifice, ta, deserves, rich, weight, bad, poor, hand, life, end

- `1030840_80583.json review=14 sentence=sentence_29`: 最后的台词画上了一个很好的句号：一切都是过眼云烟，唯独家人是永恒。
- `1049410_23830.json review=6 sentence=sentence_24`: 这就像我们现实中做梦一样，我们能记住的梦很少，可这很少的梦里却发生了非常多，时间跨度非常大的事情。
- `1049590_61216.json review=5 sentence=sentence_22`: 我一个人做不了什么，但有一口气点一盏灯，有灯就有人。

### Subtopic 23 (269 docs)

Top words: missions, mission, story missions, objective, main, misiones, story, doing, missionen, objectives

- `1012790_10355.json review=5 sentence=sentence_7`: The first part of my mission goes off without a hitch.
- `1012790_10355.json review=6 sentence=sentence_10`: But the artifact mission is what got me.
- `1012790_10355.json review=6 sentence=sentence_41`: Now, I always do artifact missions first.

### Subtopic 24 (261 docs)

Top words: rogue, rpg, roguelite, roguelike, warhammer, legacy, marauders, elements, escape, similar

- `1009290_6672.json review=7 sentence=sentence_30`: 근데 과연 이게 액션 어드벤쳐 RPG 일까요?
- `1009290_6672.json review=7 sentence=sentence_48`: 보통의 RPG 게임들이라면 무기/방어구 상점이 마주보거나 바로 옆에 있어야 한다는 기본 개념마저 버린 게임이니까요.
- `1009290_6672.json review=7 sentence=sentence_73`: 이 게임이 일반적인 RPG와는 다르다고 앞에서 언급했는데, 기본적으로 브레이크나 스킬체인 등으로 데미지 딜링을 크게 하지 않으면 동레벨에서도 고전하게끔 설계 되어 있기 때문입니다.

### Subtopic 25 (261 docs)

Top words: deck, cards, card, karten, energy, sts, qe, turn, round, draw

- `1049590_61216.json review=11 sentence=sentence_77`: To me, this game feels like a real-time card game where you "deckbuild" your route with efficient pathing then when in-game you execute your deck, playing around and knowing your opponents' (meta) routes (or decks).
- `1102190_20493.json review=3 sentence=sentence_2`: Slay the Spire turned me into a roguelike deckbuilding junkie.
- `1102190_20493.json review=3 sentence=sentence_6`: First, the deckbuilding itself.

### Subtopic 26 (260 docs)

Top words: dlc, jrpg, lessons, katana, time got, fortune, rpg, deaths, qte, aoe

- `1009290_6672.json review=7 sentence=sentence_3`: 혹시 본인이 소아온 애니메이션을 재밌게 봐서 구입을 원하신다면, 일반적인 JRPG 장르의 게임이라 생각하고 구입하시는 거라면,실망하실 각오 미리 단단히 하시길 바랍니다.
- `1009290_6672.json review=7 sentence=sentence_4`: 분명 이 게임은 소아온 원작 노벨/애니팬들에겐 괜찮은 작품입니다.
- `1009290_6672.json review=7 sentence=sentence_32`: 그리고 언뜻 보면 맵이 광활해 보이지만, 못가게 막아놓은 부분이 많아 반픈월드라고 조차 말할 수도 없습니다.4.

### Subtopic 27 (258 docs)

Top words: standalone, goh, returnal, ps2, sony, suffering, uncharted, sekiro, hearts, dps

- `1030210_11120.json review=4 sentence=sentence_27`: Самое интересно началось уже после того как я начал играть непосредственно в "игру"."Игра" буквально надменный, нахальный плевок в лицо всем фанатам этой серии.
- `1030210_11120.json review=4 sentence=sentence_29`: Как бы описать геймплей...
- `1030210_11120.json review=7 sentence=sentence_22`: Я до последнего надеялся, что ролики на движке игры – лишь временные заглушки.

### Subtopic 28 (254 docs)

Top words: ps

- `1016950_4196.json review=12 sentence=sentence_31`: Собственно слово интеллект к нему никак не относится.
- `1029780_17642.json review=6 sentence=sentence_10`: Какой профит от этого?
- `1030210_11120.json review=4 sentence=sentence_23`: У меня нет никакого другого объяснения происходящему...

### Subtopic 29 (239 docs)

Top words: ai, ki, dumb, ia, bots, bad, stupid, just bad, improved, die

- `1009290_6672.json review=5 sentence=sentence_16`: This doesn't even get into how you can customize your AI's behavior and Extra skills.
- `1016800_13368.json review=10 sentence=sentence_56`: Ja, die Bewegungen und die KI können mit Größen wie TLOU2 zum Beispiel nicht mithalten.
- `1016950_4196.json review=13 sentence=sentence_12`: - AI that makes the pitiful BB2 AI look competent.

### Subtopic 30 (238 docs)

Top words: devs, developers, update, updates, entwickler, feedback, developer, dev, die entwickler, community

- `1016800_13368.json review=3 sentence=sentence_7`: But you can tell the devs poured their hearts out into this.
- `1016800_13368.json review=6 sentence=sentence_29`: I really hope the Dev team and studio arent impacted by world events and can continue to make new material!
- `1016800_13368.json review=9 sentence=sentence_13`: man hat echte Fotos von den Entwicklern, die in Tschernobyl gemacht wurden ;)

### Subtopic 31 (236 docs)

Top words: recommend game, review, recommend, reviews, negative, negative review, review game, current state, game, state

- `1016950_4196.json review=6 sentence=sentence_9`: Currently there is nothing worth playing in this game, maybe I'll come back later, but for now, I will look into how to play fumbbl, it's ugly but at least it works.
- `1016950_4196.json review=14 sentence=sentence_16`: Ill check back in when the game is finished, but right now I cannot recommend this to anyone.
- `1029780_17642.json review=12 sentence=sentence_3`: 好き嫌い別れるジャンルなので評価がどうなるのか。

### Subtopic 32 (236 docs)

Top words: dlc, dlcs, content, upcoming, prices, base game, paid, released, price, final

- `1016950_4196.json review=5 sentence=sentence_19`: I'm still mad about paying full price for BB3 and having teams parted out to me as paid DLC.
- `1016950_4196.json review=15 sentence=sentence_8`: Et comme un fan naïf j'ai acheté la version "brutal" en me disant que je pourrais peut être avoir plus de races, de DLC, de Star Players ou de personnalisation ...
- `1030830_26263.json review=6 sentence=sentence_4`: Quanto as DLCs são três bem curtinhas (entre 2-3 horas de duração cada), tem pouca relação cm o jogo principal (qe por sua vez dura cerca de 10-12 horas) e já vem inclusas na versão definitiva.

### Subtopic 33 (227 docs)

Top words: jeu, le, et, le jeu, pas, je, pour, ne, qui, les

- `1016950_4196.json review=8 sentence=sentence_1`: Un jeu que j'attendais car en temps que pigeon assumé cela me fait toujours plaisir de dépenser mon argent pour des évolutions des jeux que j'aime.
- `1016950_4196.json review=8 sentence=sentence_2`: Le modèle économique ne me dérange pas non plus, ok il veulent rentabiliser leur jeu, ba pourquoi pas, du moment que c'est pas abusé, c'est au moins l'assurance d'avoir un suivi dans les ajouts de contenu.
- `1016950_4196.json review=8 sentence=sentence_4`: Je leur en veux même pas que le jeu soit ultra bugger et quasi injouable à la sortie après avoir repousser le jeu de deux ans.

### Subtopic 34 (222 docs)

Top words: voice, voice acting, acting, voices, lines, english, dialogue, characters, great, job

- `1030840_80583.json review=18 sentence=sentence_4`: Voice acting is great, not sure if it's the original audio just remastered or if it's re-done.
- `1034140_19644.json review=12 sentence=sentence_16`: The VA is amazing (Steve Hamm as the Captain being a particular favourite of mine).
- `1034140_19644.json review=13 sentence=sentence_6`: Voice acting in game was well done both in the chatbox part and in the video animation+ Space combat is well polished and smooth

### Subtopic 35 (215 docs)

Top words: english, translation, language, russian, translate, google, las, german, oyun, auf

- `1016800_13368.json review=7 sentence=sentence_27`: 上でも書いた通り、日本語テキストの質が高くキャラ付けがはっきり読み取れるので、会話が楽しい。
- `1016800_13368.json review=10 sentence=sentence_15`: Die Sprachausgabe habe ich auf russisch mit UT gestellt, um die Atmosphäre zu verdichten.
- `1029780_17642.json review=5 sentence=sentence_8`: Es una característica genial que mas juegos deberían implementar sobretodo juegos como este que son mas complejos que la media estándar.-Por ultimo mencionaré algo que me parece muy importante pero muchos desarrolladores de videojuegos pasan por alto o les vale gorro y es traducir al idioma materno de nosotros los jugadores, es increíble como hay juegos que venden muy bien pero no se dignaron nunca en sacar una traducción oficial a diversos id...

### Subtopic 36 (214 docs)

Top words: characters, personality, character, memorable, cast, charaktere, character development, written, depth, unique

- `1016800_13368.json review=10 sentence=sentence_52`: Die Mitglieder und Charaktere sind sehr gut geschrieben und haben durch ihre Wortwahl und Ausdrucksweise richtig Persönlichkeit.
- `1029690_19693.json review=11 sentence=sentence_16`: But the characters are a little more memorable and there is a narrative that shines through.
- `1029780_17642.json review=7 sentence=sentence_3`: The little characters have enough personality to make you care about them individually and the seasonal changes are very immersing!

### Subtopic 37 (210 docs)

Top words: ban, hackers, banned, discord, cheating, account, post, forums, reddit, community

- `1016950_4196.json review=14 sentence=sentence_12`: with new rules implemented apparently on a discord somewhere, but couldn't be bothered to even put the notification in game or on the update page.
- `1034140_19644.json review=12 sentence=sentence_30`: One final point to make, the moderator's on the Steam forums should be ashamed for censoring and closing threads that discuss the issues surrounding FOW and Subverse.
- `1044720_18239.json review=8 sentence=sentence_20`: I've played a lot of Banished so I immediately knew what to do, but other users won't.

### Subtopic 38 (209 docs)

Top words: flaws, bad game, game isnt, bad, design, game, mechanics, missing, gameplay, good game

- `1016950_4196.json review=3 sentence=sentence_19`: There is no meat to this game, apart from the new stadiums (only one is available right now of course) everything is insanely low effort and rehashed material to the point that you start to wonder what they actually DID work on.
- `1016950_4196.json review=3 sentence=sentence_50`: I’m no expert in game design but I can’t imagine messing things up this hard if I had been the creative director behind this game.
- `1029690_19693.json review=5 sentence=sentence_2`: Core elements remain the same as in previous iterations, but new mechanics are implemented very poorly and detract from the overall experience.

### Subtopic 39 (208 docs)

Top words: ai, amidst, echo, world, artificial, num, essence, shape, words, thank

- `1057090_26450.json review=14 sentence=sentence_18`: 将被污染的水变得清澈透明，将冰雪覆盖的山峰重新焕发生机，将阴暗恐怖的地洞深处重新被光芒照亮，以及帮助其他生灵安家落户，拯救每颗需要生根发芽的种子等等，它是真的通过自己的努力一点点让被腐朽侵蚀的尼文恢复生机。
- `1088850_4860.json review=11 sentence=sentence_55`: ——卡魔拉，是灭霸的女儿我晓得，其他的根本不了解。
- `1092790_122929.json review=3 sentence=sentence_64`: 凯西可能是假死，因为她不是骷髅olddata里的地点照片都和nazi有关
