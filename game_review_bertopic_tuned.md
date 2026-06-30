# Game Review BERTopic

- Generated: 2026-07-01T05:12:17
- Input H5: `C:\Users\admin\Documents\eval_array_latent\game_review_data\embedding_h5.h5`
- Corpus format: `embedding_h5`
- H5 games scanned: 2000
- H5 size: 152.912 GB
- Sample method: `balanced_prefix_per_game`
- Sampled documents: 100000
- Embedding dimension: 1024
- UMAP `n_neighbors`: 15
- HDBSCAN `min_cluster_size`: 50
- CountVectorizer `min_df`: 1
- Random state: 42
- Fit time: 85.9 seconds

Note: this run uses a deterministic per-game prefix sample and skips metadata records with `review_id < 3` before fitting BERTopic. It is a practical topic snapshot, not a full all-sentence HDBSCAN fit.

## Environment

| Package | Version |
|---|---:|
| `bertopic` | `0.17.4` |
| `hdbscan` | `0.8.44` |
| `umap-learn` | `0.5.12` |
| `h5py` | `3.15.1` |
| `numpy` | `2.4.3` |
| `scikit-learn` | `1.8.0` |

## Summary

- Topics excluding outliers: 127
- Outlier documents: 54898
- Outlier rate: 54.90%

## Topic Table

| Topic | Count | Top Words |
|---:|---:|---|
| -1 | 54898 | outliers |
| 0 | 15006 | bug, boss, npc, bossboss, boss boss, steam, npc npc, debuff, bug bug, spoiler |
| 1 | 1474 | hours, horas, hour, minutes, played, took, hours game, game hours, played hours, took hours |
| 2 | 1362 | bad, say, im, thats, know, wrong, thing, dont, just, things |
| 3 | 1244 | multiplayer, coop, online, friends, servers, play, mode, server, players, singleplayer |
| 4 | 1104 | steam, ea, steam deck, deck, launcher, epic, game steam, launch, origin, account |
| 5 | 875 | game, love game, love, playing, loved, enjoyed, like game, played, really, wanted |
| 6 | 872 | music, soundtrack, sound, sound design, sounds, sonora, audio, tracks, design, trilha |
| 7 | 702 | price, worth, sale, buy, promoo, worth price, preo, buying, espere, espere uma |
| 8 | 631 | devcats, iron lung, lung, lung iron, iron, xy, qte, cats, lowend, gestalt |
| 9 | 624 | le, et, jeu, les, pas, je, des, pour, qui, ce |
| 10 | 569 | fps, rtx, settings, gtx, dlss, gpu, ram, nvidia, cpu, pc |
| 11 | 560 | combat, combate, battles, fun, fun combat, combat fun, combat feels, game combat, feels, battle |
| 12 | 516 | dlc, dlcs, dlc dlc, base game, content, paid dlc, base, edition, paid, game dlc |
| 13 | 508 | graphics, art, pixel, grficos, pixel art, style, grafik, art style, beautiful, los |
| 14 | 473 | price, sale, worth, preo, jogo, price tag, tag, game sale, reais, game |
| 15 | 455 | la, pizza, chocolate, el, minutos, overcooked, taza, bowl, steak, step |
| 16 | 453 | car, cars, driving, drive, truck, vehicles, grip, wheel, brake, carro |
| 17 | 433 | map, maps, mapa, mapas, world map, explore, location, areas, world, map map |
| 18 | 419 | devs, developers, dev, developer, team, community, feedback, dev team, work, listen |
| 19 | 418 | controller, controls, mouse, keyboard, button, controle, xbox, buttons, press, wasd |
| 20 | 412 | review, reviews, negative, negative review, write, write review, update review, read, original review, negative reviews |
| 21 | 402 | weapons, weapon, ammo, guns, gun, rifle, shotgun, rifles, melee, shotguns |
| 22 | 395 | review, recommend, recommend game, reviews, negative, negative reviews, game, review game, positive, negative review |
| 23 | 389 | quake, house flipper, flipper, quake quake, house, champions, overcooked, nintendo, h1, spoiler |
| 24 | 385 | roguelike, rogue, doom, roguelite, rpg, souls, diablo, skyrim, dungeon, like |
| 25 | 378 | simulator, stardew, sims, stardew valley, sim, valley, walking, farming, walking simulator, simulation |
| 26 | 376 | horror, horror game, resident evil, resident, evil, survival horror, scary, survival, scares, horror games |
| 27 | 369 | translation, english, language, chinese, traduo, translate, bersetzung, bersetzt, idioma, localization |
| 28 | 361 | puzzles, puzzle, puzzle game, solving, solve, puzzle games, easy, picross, puzzles game, riddles |
| 29 | 348 | story, ending, plot, interesting, story story, end, storytelling, characters, emotional, narrative |
| 30 | 310 | enemies, enemy, attack, dodge, attacks, hit, cover, block, enemy types, enemy attacks |
| 31 | 303 | animals, cat, dogs, animal, pet, dog, cats, horse, pets, animais |
| 32 | 292 | mod, chinese, bug, milk1, tiny bunny, deeply, swipe, chinese version, adv, translated |
| 33 | 292 | early access, access, early, game early, access game, release, alpha, released, access early, access release |
| 34 | 265 | houses, build, building, buildings, village, furniture, house, bricks, placed, base |
| 35 | 262 | enjoyed, fun, enjoying, really, like, really enjoyed, enjoy, liked, enjoyed time, did |
| 36 | 260 | mods, mod, modding, daggerfall, modding community, dot, daggerfall unity, community, unity, modded |
| 37 | 251 | characters, character, main, memorable, personajes, protagonists, endearing, suffering builds, characters characters, builds character |
| 38 | 247 | ai, pathing, bots, ia, ai ai, pathfinding, ai just, ai doesnt, la ia, ai really |
| 39 | 236 | difficulty, hard, difficulties, difficult, difficulty settings, harder, levels, challenging, hard mode, challenge |
| 40 | 235 | good game, great game, good, amazing, game, great, game amazing, game great, game game, amazing game |
| 41 | 218 | really really, recommend, recommend game, highly, really, highly recommend, game recommend, game highly, game, definitely recommend |
| 42 | 210 | bugs, bug, updates, patch, patches, update, fixed, fix, minor, issues |
| 43 | 209 | quests, quest, sidequests, main, complete, main story, story quest, fetch, story, main quest |
| 44 | 207 | ship, ships, fishing, boat, fleet, boats, sea, minigame, underwater, atlantic |
| 45 | 206 | recommend, recommended, highly, recommendation, highly recommended, recommend recommend, highly recommend, dont recommend, je, recommend highly |
| 46 | 202 | pc, port, version, pc port, pc version, ps4, console, consoles, gen, ps2 |
| 47 | 198 | ai2lucy, ai2lucy got, city final, got problems, edition ai2lucy, final edition, scarlet maidenmaid, milfy city, maidenmaid mansion, problems scarlet |
| 48 | 198 | discord, banned, ban, community, bans, forums, discord server, permanently, server, post |
| 49 | 195 | rating, overall, rate, solid, overall game, nota, rate game, recommend, id, nota para |
| 50 | 193 | remaster, remake, remastered, original, remasters, el, que, original game, mafia, version |
| 51 | 193 | issues, game, mechanics, unfortunately, biggest, problem, issues game, problems game, gameplay, lot |
| 52 | 192 | vr, vr games, vr game, best vr, vrchat, headset, vr vr, cvr, best, games |
| 53 | 188 | cards, card, deck, decks, deckbuilding, draw, cards deck, trading cards, card games, hand |
| 54 | 188 | crafting, craft, inventory, materials, item, items, resources, storage, gear, craft item |
| 55 | 187 | missions, mission, complete, misiones, missions missions, rewards, misses, campaign, missions feel, main missions |
| 56 | 183 | favorite, best, favorite games, played, favorite game, favourite, games, best games, ive played, best game |
| 57 | 172 | voice, voice acting, acting, actors, voice actors, voices, voiced, acted, characters, voice acted |
| 58 | 168 | issues, problems, problem, issue, needs, things, algumas, lot work, maybe lot, foram |
| 59 | 164 | skills, skill, skill points, points, level, character, tree, levels, gear, passive |
| 60 | 159 | bosses, boss, boss fights, fights, final boss, fight, boss fight, final, challenging, bosses game |
| 61 | 158 | money, dinheiro, waste, waste money, sobrando, se tiver, dinheiro sobrando, compre se, debt, tiver dinheiro |
| 62 | 149 | save, saves, save game, save file, autosave, file, saved, progress, manual, manual save |
| 63 | 146 | plant, crops, grow, crop, farming, plants, ready, harvest, seeds, lets |
| 64 | 142 | speed limit, limit, speed, locomotive, railway, lcd, car, ui, signal, arrow |
| 65 | 135 | good decent, bad, decent bad, beautiful good, good, decent, beautiful, bad beautiful, bad bad, schlecht |
| 66 | 135 | needs, hope, future, suggestions, potential, game, forward, things need, game potential, game better |
| 67 | 134 | abandoned, updates, development, devs, update, developers, developer, game, players, dev |
| 68 | 133 | dishes, cooking, cleaning, customers, food, clean, toilet, restaurant, ingredients, flip |
| 69 | 132 | humor, jokes, funny, laugh, humour, joke, make laugh, hilarious, silly, references |
| 70 | 123 | bugs, bugs game, game breaking, bug, gamebreaking, buggy, breaking, issues, breaking bugs, mnh |
| 71 | 121 | achievements, achievement, score, leaderboardsranks isnt, high score, care leaderboardsranks, leaderboardsranks, isnt necessary, necessary progress, achievements wait |
| 72 | 121 | animations, animation, animaes, animated, smooth, theyre, animations bit, animations look, sprites, stiff |
| 73 | 121 | human fall, oa, fall flat, oo, flat, fall, human |
| 74 | 120 | art, art style, style, arte, artistic, visuals, artstyle, beautiful, painting, work art |
| 75 | 119 | check run, run paint, paint, requirements check, check, pc requirements, requirements, teens, kids teens, teens adults |
| 76 | 119 | wa, open eyes, ga, eyes, logic open, wo suru, suru, wo, pole, pal |
| 77 | 118 | ui, interface, menus, ui ui, menu, ui elements, sampai, user interface, dalam, user |
| 78 | 118 | xcom, xcoms, xcom2, reboot xcoms, reboot, like xcom, phoenix point, strategy, phoenix, tactics |
| 79 | 112 | nasa, nasa spare, ask nasa, spare computer, spare, computer, ask, computer difficulty, just press, press |
| 80 | 112 | dark souls, dark, souls, grind, souls grind, difficult dark, grind grind, difficult, grind difficult, grind dark |
| 81 | 106 | pvp, pve, pvp pvp, pve pvp, pvp f2p, server, gzw, consegu, pvpve, pvp game |
| 82 | 96 | fnaf, birthday, stayed, witch, fish, happy, came, nice, love, gameplay |
| 83 | 94 | fell, death, die, died, dead, lyorum, ento, burried, exato momento, exato |
| 84 | 90 | lumina, nazi, beloved, award, start new, porn, wife, buy game, thank, thought |
| 85 | 90 | chinesewe, chinesewe need, need chinesewe, need, need chinese, chinese, chinese need, besoin, chinois, avons besoin |
| 86 | 90 | demons, werewolf, named, night shifts, protagonist, kyle, amanda, curse, ritual, job |
| 87 | 86 | hard master, learn hard, easy learn, master, master easy, learn, easy, hard, aprender, fcil |
| 88 | 85 | free, free play, game free, freetoplay, free game, play, game freetoplay, base game, game, f2p |
| 89 | 82 | mullet madjack, mason lindroth, mullet, madjack, flatout, mason, carnage, limbo, upd, god eater |
| 90 | 80 | simple, easy, game simple, gameplay, gameplay simple, simple game, loop, gameplay loop, mechanics, complex |
| 91 | 80 | bugs, experienced, issues, optimization, crashes, optimized, glitches, havent experienced, encountered, crash |
| 92 | 79 | metroidvania, metroid, metroidvanias, megaman, metal gear, genre, titulo, hollow knight, metal, msx |
| 93 | 78 | release franchise, franchise release, franchise, release, shrine gain, offer shrine, shrine, gain offer, gain, offer |
| 94 | 77 | keyboard, player keyboard, escadd, escadd player, xbox, player, pc, emmm, keyboard player, assign keyboard |
| 95 | 77 | oo, icebreakerlust, touchthe, powerliving kanazawarosewater, kanazawarosewater manor88name88s, icebreakerlust powerliving, manor88name88s triangle, manor88name88s, kanazawarosewater, milky touchthe |
| 96 | 77 | infinity, infinity price, long infinity, average long, price, average, calculadora, com pilha, uma calculadora, pilha |
| 97 | 75 | dry instead, watch paint, paint dry, dry, paint, watch, midnight paradise, instead watch, midnight, paradise |
| 98 | 74 | trailer, trailers, sequel, series, fan series, srie, franchise, game look, series id, waiting years |
| 99 | 74 | campaign, campanha, campaigns, imperialism, kampagne, just campaign, campaign does, endgame content, verschiedene, scripted |
| 100 | 73 | developers, dev, developer, devs, developed, person, game, game developed, studios, small |
| 101 | 72 | level design, design, level, levels, designed, design good, game level, games level, design bit, great level |
| 102 | 72 | crashes, crash, crashing, game crashes, game crash, crashed, crash game, random, screen, constantly crashes |
| 103 | 71 | camera, angle, view, camera angle, camra, screws, cameras, fixed camera, angles, camra est |
| 104 | 68 | platforming, platformer, platformers, sections, platforming sections, puzzles, challenging, platformer sections, good platformer, fights |
| 105 | 67 | madrid, und ich, spieler, und, mick, mannschaft, meine mannschaft, der, nicht weg, caej |
| 106 | 65 | environment, environments, atmosphere, beautiful, scenery, etkileim, atmosphere just, ne ok, vistas, stunning |
| 107 | 65 | p5, djmax, psp, ps5, reborn, ps3, enhanced, persona, ps, version |
| 108 | 65 | winning son, winning, psychopath, psychopath psychopath, son winning, son, ya winning, ya, sir, stoic |
| 109 | 65 | total war, total, war, rome, rts, war games, shogun, warhammer, strategy, war game |
| 110 | 63 | tutorial, tutorials, das tutorial, das, habe ich, los tutoriales, content doing, tutorials tutorial, tutorial single, tutoriales |
| 111 | 62 | graphics forget, forget reality, reality, reality graphics, forget, graphics, voc esquece, esquece, esquece que, que realidade |
| 112 | 61 | little nightmares, northern journey, northern, nightmares, journey, npc, little, games |
| 113 | 59 | replayability, replay, replay value, value, replayable, challenges, good replayability, difficulty, different, add |
| 114 | 58 | janken, isekai janken, simulator fusion, isekai, janken hero, fusion, hero simulator, fusion isekai, hero, simulator |
| 115 | 58 | cruub, cruub cruub, king cruub, cruub king, monke, king, wren, god, jr, ya |
| 116 | 58 | gmod, gmod gmod, gb, gb gb, exe gmod, gmod exe, exe, g50100, g50100 g100200, gb gmod |
| 117 | 58 | jump, jumps, jumping, wall, fall, crystal arrow, momentum, double, slide, walls |
| 118 | 57 | bug bug, bug, dedsec, bgm bug, bug bugbug, game joke, fk, game strongly, preferably, fh |
| 119 | 56 | chinese need, need chinese, chinese, pleasekorean, pleasekorean pleasekorean, need, china, republic, scenario, war ii |
| 120 | 56 | visual, visual novel, novel, novels, visual novels, digimon, musicals, novel game, story, great visual |
| 121 | 55 | character, notre, create character, customize, mask, customize character, body, gender, customization, changer |
| 122 | 54 | significant brain, brain usage, brain, usage, significant, usage significant, uso significativo, significativo, uso, significativo crebro |
| 123 | 53 | fun game, fun, really fun, game fun, lot fun, game lot, damn fun, game damn, genuinely fun, extremely fun |
| 124 | 53 | buy game, worth, game worth, buy, pena, vale, thinking buying, comprar, definitely worth, game definitely |
| 125 | 52 | eargasm good, bad bad, eargasm, good bad, bad, audio eargasm, dont audio, good good, audio, just dont |
| 126 | 52 | rich ladys, ladys slave, slave role, leaderboardsranks, care leaderboardsranks, ladys, role play, slave, rich, role |

## Topic Examples

### Topic 0 (15006 docs)

Top words: bug, boss, npc, bossboss, boss boss, steam, npc npc, debuff, bug bug, spoiler

- `1009290_6672 review=3 sentence=sentence_2`: 游戏近期更新，这下可以彻底放弃这款粪作了~
- `1009290_6672 review=3 sentence=sentence_3`: 弃掉之前，做做好事，还是说说原因避免别人被坑吧~
- `1009290_6672 review=3 sentence=sentence_4`: 游戏近期更新了一个【极】难度的迷宫，里面的BOSS高达破天荒的210级，血量高达1亿多，然而我们角色还是50级。
- `1009290_6672 review=3 sentence=sentence_5`: 估计万代这神奇公司的脑回路，是觉得平时那些高难度BOSS都被秒秒秒了，所以出了一个新的难度让我们可以继续刷，同时副本产出属性更高的【新装备】。
- `1009290_6672 review=3 sentence=sentence_6`: 然而，这些新装备只是换了个颜色，模型基本大同小异，基本毫无诚意。
- `1009290_6672 review=3 sentence=sentence_11`: 也就是不管你其他属性多高，你的门槛达不到那个数值，你的攻击会全部MISS。
- `1009290_6672 review=3 sentence=sentence_12`: 这就导致，你更新了新难度，那我们现在输出全MISS，请问我们拿什么去打？
- `1009290_6672 review=3 sentence=sentence_13`: 第一，把装备全部换了，主力去提高命中，放弃其他属性。
- `1009290_6672 review=3 sentence=sentence_14`: 输出能命中，但是刮痧刮到吐你一个游戏，做成这个样子？
- `1009290_6672 review=3 sentence=sentence_16`: 第二，开科技去刷这个副本，然后刷出来一套装备了，我们同时增加了命中和其他属性，终于可以关掉科技正常的玩游戏咯~

### Topic 1 (1474 docs)

Top words: hours, horas, hour, minutes, played, took, hours game, game hours, played hours, took hours

- `1000360_6765 review=3 sentence=sentence_16`: There are hardly ANY layers on top of the base mechanics of the game, yet I still feel like I'm barely scratching the surface of my main character, Marie, after 100 hours of playtime.
- `1000410_2404 review=3 sentence=sentence_15`: Each level will take anywhere from 25 minutes to nearly an hour to complete.
- `1009560_6684 review=5 sentence=sentence_2`: I’ve put in some hours to really try out how I feel and here’s my thoughts:
- `1012790_10355 review=6 sentence=sentence_13`: after doubling my playtime up to 57 hours:
- `1013320_5961 review=3 sentence=sentence_1`: In 5 minutes of gameplay, I came across several red flags.
- `1013320_5961 review=3 sentence=sentence_4`: Yes, within the first 5 minutes, the game shows you the shop.
- `1013320_5961 review=3 sentence=sentence_6`: Number 2 : The very first arbitrary timer (that the game desperately tries to hide by immediately shoving you forward back to gameplay), is 8 hours.
- `1013320_5961 review=3 sentence=sentence_7`: Number 3 : The game "gave me" 50 gems and immediately tried to get me to spend them on skipping a 30 minute arbitrary timer.
- `1013320_5961 review=3 sentence=sentence_8`: That's when I closed down the game to wait out the 30 minute timer instead, which I had to do outside the game.
- `1013320_5961 review=3 sentence=sentence_9`: All that in 5 minutes.

### Topic 2 (1362 docs)

Top words: bad, say, im, thats, know, wrong, thing, dont, just, things

- `1000410_2404 review=3 sentence=sentence_37`: Oh, how wrong we all were.
- `1001270_4904 review=4 sentence=sentence_5`: Ama şuan en önemli olması gereken olay istasyon olayının silinip kendimizin götürme olayı gelmeli.
- `1001270_4904 review=5 sentence=sentence_21`: I can't explain it, it just doesn't feel natural.-
- `1003590_7381 review=3 sentence=sentence_13`: I wouldn't have it any other way.
- `1013320_5961 review=3 sentence=sentence_17`: You should know, you put it there.
- `1015500_833 review=7 sentence=sentence_2`: very frustrating !!!!!
- `1016950_4196 review=5 sentence=sentence_9`: What follows is a modest selection from the plethora of reasons why.•
- `1016950_4196 review=5 sentence=sentence_28`: What was the logic behind that?
- `1016950_4196 review=5 sentence=sentence_47`: Honestly, who made that?
- `1020790_3591 review=5 sentence=sentence_4`: Será simplesmente frustrante!

### Topic 3 (1244 docs)

Top words: multiplayer, coop, online, friends, servers, play, mode, server, players, singleplayer

- `1000360_6765 review=3 sentence=sentence_35`: With all of that said, I MUST WARN YOU, THERE IS NO NATIVE ONLINE FOR THIS GAME.
- `1000360_6765 review=3 sentence=sentence_36`: THAT DOES NOT MEAN THAT YOU CANNOT PLAY HELLISH QUART ONLINE.
- `1000360_6765 review=3 sentence=sentence_37`: This is a major, ongoing topic in the community still, but Kubold has stated before that developing native online play for Hellish Quart is a bit out of scope currently.
- `1000360_6765 review=3 sentence=sentence_39`: This is obviously a sub-optimal solution for playing online, but works surprisingly well, especially if the two players are within the same country/time zone.
- `1000360_6765 review=3 sentence=sentence_41`: If you're still wary about the quality of online play, find someone that owns Hellish Quart and ask if they can allow you to test drive the game with them via Remote Play or Parsec!
- `1001270_4904 review=4 sentence=sentence_17`: Lütfen oyunu salmayın geliştirmeye devam edin başarılar diliyorum.-hayatta kalma öğeleri karakterin açlık susuzluk temizlik tarzı etmenler-Coop girildiğinde belli levelden sonra arkadaşım bulaşıkları yıkayamıyor.
- `1001270_4904 review=4 sentence=sentence_18`: Sadece oyunu kuran kişi yıkıyor.-Yardımcılar duvar içine giriyor.-Oyunda fare, sinek, böcek gibi elementlerin olması oyunu daha gerçekci yapabilir.
- `1001270_4904 review=5 sentence=sentence_4`: And...- You get to play with your friends!
- `1003590_7381 review=5 sentence=sentence_30`: A ranked coop online gameplay?
- `1003590_7381 review=6 sentence=sentence_6`: コネクテッドになって対人やco-opをクロスプレイで出来るようになりました。

### Topic 4 (1104 docs)

Top words: steam, ea, steam deck, deck, launcher, epic, game steam, launch, origin, account

- `1003590_7381 review=3 sentence=sentence_1`: The playtime I have on Steam is nowhere close to the amount of time I've put into this game on Xbox.
- `1009290_6672 review=3 sentence=sentence_1`: 艾玛，今天呢，游戏发行不到三个月就开始打折咯~
- `1022310_1594 review=4 sentence=sentence_20`: There aren't even Steam achievements and although not everyone needs them, I like achievements simply from the feeling of progression.
- `1022980_1451 review=4 sentence=sentence_15`: 在目前的EA阶段（20年9月11日），本游戏离它的完全体还有较长的一段距离。
- `1030840_80583 review=5 sentence=sentence_3`: Don't run this game directly of steam or the launcher, instead find the install directoryX:\SteamLibrary\steamapps\common\Mafia Definitive Edition
- `1030840_80583 review=5 sentence=sentence_5`: click on mafiadefinitiveedition.exeProperties > compatibility > click ON the checkbox 'Run this program as an administrator'.
- `1030840_80583 review=5 sentence=sentence_6`: Once in the game, open settings, and set 'use launcher' to OFF.
- `1034140_19644 review=6 sentence=sentence_14`: Not right now unless you you really like hentai games and just want to support them on steam so they don't get censored/banned.
- `1036240_1798 review=4 sentence=sentence_7`: TL;DR – As I would only purchase this game on sale, and Steam lacks a middle ground option,
- `1037020_2760 review=3 sentence=sentence_43`: 鉴于《ScourgeBringer》还在EA阶段，而这些属于可以慢慢填充的内容，我倒也不是很担心它未来的表现就是了。

### Topic 5 (875 docs)

Top words: game, love game, love, playing, loved, enjoyed, like game, played, really, wanted

- `1000030_2098 review=3 sentence=sentence_34`: For the first few regions of the game I stayed on Chill and didn't feel like I was 'wussing out' of the 'real' experience at all.
- `1000360_6765 review=5 sentence=sentence_1`: I've fenced for years before I was injured and couldn't do it anymore, this game really made me feel like I was back on the piste.
- `1003590_7381 review=3 sentence=sentence_2`: When I first heard of Tetris Effect back in 2018, I was so mesmerized by the visuals the game had.
- `1003590_7381 review=3 sentence=sentence_6`: I'm not exaggerating when I say that buying this game actually changed my life.
- `1003590_7381 review=4 sentence=sentence_2`: I can finally get him to use it fairly willingly as the game has nice music and good visuals.
- `1009560_6684 review=4 sentence=sentence_9`: I still loved playing you.
- `1009560_6684 review=5 sentence=sentence_1`: I enjoy the game a lot.
- `1012790_10355 review=6 sentence=sentence_1`: I feel like personal experiences are some of the best ways to explain the appeal of a game to someone.
- `1016800_13368 review=3 sentence=sentence_23`: I just finished the game, and I can confidently say it was all worth it.
- `1016800_13368 review=4 sentence=sentence_2`: I have to say that I wasn't sure what to expect from this game when I purchased it.

### Topic 6 (872 docs)

Top words: music, soundtrack, sound, sound design, sounds, sonora, audio, tracks, design, trilha

- `1000030_2098 review=3 sentence=sentence_21`: I was surprised to find that the soundtrack was composed by the same artist as the previous games since the step up in production and general quality is incredible.
- `1000030_2098 review=3 sentence=sentence_22`: CSD and CSD2 had good soundtracks to be sure, but CSD3 has bops that are stuck in my head and occupy my Spotify playlists.
- `1000030_2098 review=3 sentence=sentence_26`: You're saved from the potential hearing damage that everything going off at once would cause since it quickly 'scrolls' through the available dishes for an intensely satisfying series of chimes.
- `1000410_2404 review=3 sentence=sentence_27`: This is unfortunate, as it was composed by Andrew Hulshult, the de facto boomer shooter OST guy.
- `1003590_7381 review=6 sentence=sentence_4`: ただ、このテトリスエフェクトでは動かしたりドロップしたり消したりした時に、背景世界と音楽との一体感を強化しており、より一層トリップできるような作りになっています。
- `1003590_7381 review=6 sentence=sentence_8`: VRでプレイするとサラウンドな音声に加え、背景が立体空間になるので没入感は更に増加します。
- `1016790_1893 review=3 sentence=sentence_6`: Simplistic and dark soundtrack that adds another layer to the gloomy and dark environments•
- `1016790_1893 review=3 sentence=sentence_42`: The soundtrack is simplistic and adds a sense of mystery of awareness to every area you explore.
- `1016790_1893 review=3 sentence=sentence_45`: Every new area you explore is dark and grim which creates this fantastic atmosphere along with the soundtrack.
- `1017900_1112 review=4 sentence=sentence_6`: - La musique est vraiment sympa et bien remasterisée et les bruitages sont bons également

### Topic 7 (702 docs)

Top words: price, worth, sale, buy, promoo, worth price, preo, buying, espere, espere uma

- `1000360_6765 review=3 sentence=sentence_31`: PROMO AND ADS ON STAGES!
- `1000410_2404 review=3 sentence=sentence_2`: Was it worth the wait?
- `1009560_6684 review=5 sentence=sentence_23`: I get it, we want nice things, however, cheap items doesn’t mean it’s ugly.
- `1009560_6684 review=5 sentence=sentence_27`: Some cheap, some expensive, whatever.
- `1015500_833 review=5 sentence=sentence_8`: For the price, this is almost a scam, especially since this has been released for over a year.
- `1022310_1594 review=4 sentence=sentence_29`: It feels like I didn't get what I paid for.
- `1022310_1594 review=4 sentence=sentence_33`: It doesn't seem like they tested their pricing model as I'm surprised at this current approach.
- `1029550_11568 review=3 sentence=sentence_29`: It's free!☐ Worth the price☐ If it's on sale☐ If u have some spare money left☐ Not recommended☐ You could also just burn your money---{ 🤪Bugs🤪 }---☐ Never heard of☐
- `1034140_19644 review=3 sentence=sentence_7`: 펀딩의 마음으로 구매하실 의향이 있다면 투자하셔도 좋을 수준입니다[우리 모두가 원하는 중요한 컨텐츠]-
- `1036240_1798 review=4 sentence=sentence_4`: Picking it up on sale was fine if you can deal with all the mediocrity.

### Topic 8 (631 docs)

Top words: devcats, iron lung, lung, lung iron, iron, xy, qte, cats, lowend, gestalt

- `1030210_11120 review=5 sentence=sentence_13`: Посреди всего этого с гомерическим хохотом бегали игроки.
- `1030210_11120 review=5 sentence=sentence_15`: Т.е. первый игрок зачищал локацию, а те кто попадал туда позже просто бегали по пустой.
- `1030210_11120 review=5 sentence=sentence_35`: По пути тебя неуверенно будут покусывать абсолютно одинаковые монстры.
- `1038250_6432 review=5 sentence=sentence_9`: Вот практически калька, если не брать во внимание парочку "новинок" в виде гонок на льду и первопроходца.
- `1041720_6700 review=7 sentence=sentence_13`: Убивать мобов тут можно задорно и весело, а этим способны похвастаться далеко не все РПГ, даже современные.
- `1041720_6700 review=7 sentence=sentence_14`: • Из того, к чему можно придраться: да, локации коридорные.
- `1041720_6700 review=7 sentence=sentence_20`: Так что стимул обойти все локу от и до есть.
- `1045720_3045 review=4 sentence=sentence_4`: Головною героїнею цієї історії тепер стає Пак Міна — ліпша подруга головного героя першої частини Йонхо.
- `1045720_3045 review=4 sentence=sentence_7`: Міна так само має втікати/ховатися/освітлювати собі шлях запальничкою (ліхтарика тепер немає), але ми отримали систему швидких команд для перевірки успішності сховку.
- `1045720_3045 review=4 sentence=sentence_9`: Збільшилася кількість ворожих сутностей і подій, які можуть нас убити або скалічити.

### Topic 9 (624 docs)

Top words: le, et, jeu, les, pas, je, des, pour, qui, ce

- `1017900_1112 review=4 sentence=sentence_3`: J'avais aimé l'ambiance à l'époque et j'aurais grandement apprécié pouvoir m'appuyer sur cette édition pour le faire en long en large et en travers aujourd'hui.
- `1017900_1112 review=4 sentence=sentence_4`: Pas possible en l'état actuel du jeu.
- `1017900_1112 review=4 sentence=sentence_5`: Points positif : - Graphiquement c'est honnête et ça dénature pas le jeu d'origine en étant bien plus regardable.
- `1017900_1112 review=4 sentence=sentence_7`: Poins négatifs : - C'EST QUOI CE PATHFINDING DÉGUEULASSE EN 2019 SÉRIEUX ?
- `1017900_1112 review=4 sentence=sentence_8`: Les unités prennent trop de place individuellement, elles se gênent et elle ne se mettent pas en formation (!!).
- `1017900_1112 review=4 sentence=sentence_10`: - Pas de portes pour les murs - l'IA est aux fraises et beaucoup trop violente en difficulté supérieure
- `1017900_1112 review=4 sentence=sentence_11`: C'est d'autant plus dommage que tout ceci est bon dans la DE du 2 donc vraiment incompréhensible.
- `1061910_14713 review=4 sentence=sentence_2`: Je l'attendais du coin du nez ce Metal : Hellsinger, et déjà à l'époque de la démo je me suis dit que cet hybride était ce qu'il me manquait dans la bibliothèque Steam.
- `1061910_14713 review=4 sentence=sentence_4`: Vous incarnez l'inconnue, démone vénère contre les enfers puisque le Diable lui a subtilisé sa chose la plus précieuse : sa voix.
- `1061910_14713 review=4 sentence=sentence_5`: Et l'inconnue, c'est pas la petite Sirène de Disney qui veut rencontrer son prince charmant, c'est une casseuse de gueules qui défouraille sec en tranchant tout ce qui est à sa portée.

### Topic 10 (569 docs)

Top words: fps, rtx, settings, gtx, dlss, gpu, ram, nvidia, cpu, pc

- `1001270_4904 review=5 sentence=sentence_10`: In the middle of playing a friend asked for help with something, I couldn't leave the game open while doing this without hearing my PC fans working hard.-
- `1003590_7381 review=4 sentence=sentence_3`: However, the game seems to make my computer run a bit too hard for the graphics that it displays, so it doesn't seem to have the best optimization.
- `1015500_833 review=5 sentence=sentence_5`: The game suffers huge framerate drops even though I meet recommended specs.
- `1015500_833 review=5 sentence=sentence_7`: no problem, yet somehow this unoptimized garbage uses 100% of my gpu.
- `1017900_1112 review=6 sentence=sentence_2`: However... I and many other users have found the FPS of the game to have appalling.
- `1017900_1112 review=6 sentence=sentence_4`: It should run well on most major graphics cards.
- `1017900_1112 review=6 sentence=sentence_5`: It seems (I'm not alone here) the game runs very poorly on GPU's more than capable for running the game.
- `1017900_1112 review=6 sentence=sentence_6`: 20-30 fps on a 1080ti is clearly not implemented well.
- `1020790_3591 review=3 sentence=sentence_13`: There's nothing improved upon graphics wise either and I actually got some stuttering here and there.
- `1034140_19644 review=3 sentence=sentence_13`: (1060으로토 4k 돌아감)[그 컨텐츠의 취향]-

### Topic 11 (560 docs)

Top words: combat, combate, battles, fun, fun combat, combat fun, combat feels, game combat, feels, battle

- `1000360_6765 review=3 sentence=sentence_2`: To start, I'm an epee fencer of ~8 years now, and like you may have read in the other comments, this game feels VERY similar to fencing in real life.
- `1000360_6765 review=4 sentence=sentence_3`: Tthe fencing itself is fluid, snappy and, most importantly, fun.
- `1034140_19644 review=6 sentence=sentence_5`: hell and grid combat are basic, but they aren't the main draw for the game anyway-
- `1035120_4085 review=3 sentence=sentence_6`: Yes, game feels clunky at times, some mechanics could've been done better or have been polished more - combat for example.
- `1035120_4085 review=7 sentence=sentence_8`: El gameplay está bastante bueno, cuenta con sistema de combate, sigilo (básico, simplemente agacharte para pasar sin ser detectado) y otras mecánicas como son los "poderes".
- `1036890_4908 review=3 sentence=sentence_11`: The combat is a lot more barebones than the previous two games, and due to streamlining, Lo Wang is reduced to a knucklehead with no chi bar or chi powers activated using key combos.
- `1036890_4908 review=3 sentence=sentence_20`: The combat plays like the Orbs of Masume challenges from SW2- a dodge fest and bullet hell spam.
- `1041720_6700 review=7 sentence=sentence_12`: Динамичная нон-таргет боевка.
- `1043260_3322 review=4 sentence=sentence_2`: What put this firmly in the no category is that pretty consistently in later game fights some of your gladiators will just stop doing anything, ruining your strategy.
- `1058650_7389 review=3 sentence=sentence_14`: At the moment you're probably going to play this game primarily in trenches, if possible, exactly because of the focus on melee combat.

### Topic 12 (516 docs)

Top words: dlc, dlcs, dlc dlc, base game, content, paid dlc, base, edition, paid, game dlc

- `1009560_6684 review=4 sentence=sentence_11`: The Pets DLC launched, and your flaws grew.
- `1013320_5961 review=5 sentence=sentence_7`: 对于游戏的DLC氪金插件，我可以很负责的告诉你一句，买不买一点不影响游戏，基本都是一些头像什么的，当然你氪金，在里面购买一些其他物品，如钻石什么的，之后快速结束研究的等待时间，也是可以的，但氪金消耗也是巨大的！
- `1020790_3591 review=3 sentence=sentence_2`: It's basically a DLC which adds ~10 new characters to Storm 4 along with Boruto skins for a lot of the cast which they count as "new" characters with out there being anything new added to them.
- `1020790_3591 review=5 sentence=sentence_7`: Eles tiveram 8 anos para entregar algo no mínimo decente, e conseguiram a proeza de entregar uma DLC do Naruto Storm 4 piorada, como isso é possível?
- `1029550_11568 review=4 sentence=sentence_5`: half of the cars are DLC and the DLC cars are better than the normal ones but i can understand, after all, the game is free.
- `1030830_26263 review=5 sentence=sentence_9`: It works fine in the DLCs.
- `1038250_6432 review=3 sentence=sentence_4`: We were supposed to get everything in the so-called deluxe edition, but then they released the super deluxe edition.
- `1038250_6432 review=3 sentence=sentence_5`: Well, here they renamed the amplified edition to "year one" edition, which means they're saving the bigger expansions, add-ons for later.
- `1038250_6432 review=3 sentence=sentence_8`: Also, "Year 1 Edition players get access to all DIRT 5 content in the 12 months after launch".
- `1041720_6700 review=6 sentence=sentence_10`: Agora, se você nunca jogou o jogo original, aí sim eu digo a você para comprar este que já vem com todas as dlcs e tem as pedras bonitinhas.

### Topic 13 (508 docs)

Top words: graphics, art, pixel, grficos, pixel art, style, grafik, art style, beautiful, los

- `1016790_1893 review=3 sentence=sentence_44`: The one thing this game has going for it is the art style, it definitely fits a dark wild west purgatory from the way that your character is designed to the enemies that you encounter.
- `1017900_1112 review=5 sentence=sentence_4`: Pros: - Upgraded graphics and it looks amazing! - Some stuff are really made well and villagers can now walk on farms and are a bit smarter than in the original
- `1017900_1112 review=6 sentence=sentence_1`: This is the classic AofE game, with some lovely Hi-Res texture graphics and some great quality of life features.
- `1017900_1112 review=7 sentence=sentence_4`: У цьому перевиданні досить пристойно перемалювали всю графіку.
- `1035120_4085 review=7 sentence=sentence_6`: Su punto más fuerte es la historia y en cierto punto lo gráfico, ¿por qué en cierto punto?
- `1035120_4085 review=7 sentence=sentence_7`: Los gráficos de los personajes es muy feo, plastilina a comparación de los gráficos que tienen los objetos.
- `1038250_6432 review=4 sentence=sentence_7`: Positivos:-Os gráficos são bons.
- `1038250_6432 review=5 sentence=sentence_10`: Итак.Графика в игре отвратительна.
- `1045720_3045 review=3 sentence=sentence_1`: Aynı gerilim, aynı korku fakat daha iyi grafikler, daha iyi oynanış ve daha zor!
- `1051690_3450 review=3 sentence=sentence_30`: How about the graphics?

### Topic 14 (473 docs)

Top words: price, sale, worth, preo, jogo, price tag, tag, game sale, reais, game

- `1015500_833 review=3 sentence=sentence_3`: If you see it in promo it might be worth it to try, but don't expect a great game, just an
- `1015500_833 review=9 sentence=sentence_4`: Flash forward to around christmas time in 2020 when I saw that I was able to get the game for just 10 bucks off of sale.
- `1016800_13368 review=3 sentence=sentence_15`: Either way, this is a great game on its own, especially for the price.
- `1020790_3591 review=5 sentence=sentence_2`: Acho que posso dizer sem pensar duas vezes que este jogo é um fracasso total, e você não deveria comprar esse jogo pelo preço cheio, na verdade irei mais longe.
- `1020790_3591 review=5 sentence=sentence_16`: É bizarro, pegar uma imagem e esticá-la para um jogo que custa 230 reais.
- `1022310_1594 review=4 sentence=sentence_32`: Speaking of payment, and maybe not all people would agree with this, but personally I would have made the game free to play with one or two warbands included then allow people to buy their chosen warbands as DLC.
- `1036240_1798 review=4 sentence=sentence_6`: I would never pay full price for this game in the current state it is in.
- `1036890_4908 review=3 sentence=sentence_37`: If it goes on sale for atleast half price and you buy it expecting to play a lower budget copycat version of DOOM, go ahead.
- `1038250_6432 review=4 sentence=sentence_3`: Enfim, estava querendo comprar o Dirt 5 a muito tempo, ainda mais por ter a participação da o James e Nolan da Donut Media, e finalmente entrou em promoção, pois não estava muito afim de pagar o preço cheio pois já tinha visto muitas criticas ao jogo.
- `1040070_2071 review=3 sentence=sentence_16`: On the negative side, this is a Flash game (or really resembles one) and thus is a bit over-priced for the value.

### Topic 15 (455 docs)

Top words: la, pizza, chocolate, el, minutos, overcooked, taza, bowl, steak, step

- `1000030_2098 review=3 sentence=sentence_1`: While Cook, Serve, Delicious! was already one of my favourite games, small things about it left me wanting more.
- `1000030_2098 review=3 sentence=sentence_4`: I was incredibly hyped for Cook, Serve, Delicious!
- `1000030_2098 review=3 sentence=sentence_15`: When I saw Cook, Serve, Delicious!
- `1001270_4904 review=3 sentence=sentence_3`: nothing is cooked to order, I pre-cook everything, every dish is sitting in my freezer and goes into the microwave before serving.
- `1029550_11568 review=5 sentence=sentence_6`: соль щепоткасода 0,5 ч.л. и 1 ч.л. уксуса, чтобы её погасить
- `1029550_11568 review=5 sentence=sentence_7`: рафинированное растительное масло
- `1029550_11568 review=5 sentence=sentence_8`: 2 — 4 ст.лпшеничная мука в/с 3 стакана (объём стакана 200 мл)
- `1036240_1798 review=3 sentence=sentence_21`: In the end, I personally like it, but I'd recommend giving it some more time in the oven to flesh out the major features.
- `1036240_1798 review=3 sentence=sentence_22`: Right now it's a very solid foundation, it's good but it's not crispy yet.
- `1110910_8180 review=3 sentence=sentence_2`: ~▢ izi Pizi▢ Facilillo▢ Normal🟥 Ta Hardcore▢ Mas Dificil que pellizacar un espejo~ PUBLICO ~▢ Re Nintendo va para todos▢ pa los Pibitos▢ Ya mas Grandecitos va▢

### Topic 16 (453 docs)

Top words: car, cars, driving, drive, truck, vehicles, grip, wheel, brake, carro

- `1029550_11568 review=6 sentence=sentence_10`: Real technique affects the car.
- `1029550_11568 review=6 sentence=sentence_11`: i.e. using the foot brake to get closer to walls, clutch kicks, etc.-
- `1029550_11568 review=6 sentence=sentence_12`: Good car list, great track list.
- `1029550_11568 review=6 sentence=sentence_14`: Nice to see a bunch of real-life manufacturers, companies and cars in the game.
- `1030830_26263 review=5 sentence=sentence_21`: Now that the mountains have been made more rocky and rocks are way closer to you up on the roads, they really need to be sharper because they're so visible.
- `1038250_6432 review=4 sentence=sentence_12`: Nem vou me aprofundar falando que lama e terra seca tem o mesmo grip, e que o gelo não escorrega como deveria, mas sim por tudo que você encosta é feito de pedra, até as telas de plastico laranja tem física de rocha;-Meu deus que coisa horrorosa pilotar os RWD.
- `1038250_6432 review=4 sentence=sentence_13`: Eles tentam fazer o drift sozinho, corrigindo automaticamente.
- `1038250_6432 review=4 sentence=sentence_14`: Sério, como eu posso jogar um jogo de rali, que o carro tem que ser totalmente solto pra contornar direito, sendo que o carro não fica solto?
- `1038250_6432 review=4 sentence=sentence_16`: Os FWD e AWD também são péssimos de controlar;-NÃO TEM RALLY, como assim???
- `1063420_3769 review=3 sentence=sentence_5`: V O I D D R I V E S Y N C !-

### Topic 17 (433 docs)

Top words: map, maps, mapa, mapas, world map, explore, location, areas, world, map map

- `1012790_10355 review=3 sentence=sentence_3`: You entered the zone from a single point, and could go anywhere on the map to explore and loot.
- `1012790_10355 review=3 sentence=sentence_18`: As a student in game design myself, I can attest that is NO small task to do, throwing out months of work building, testing, and refining an ENTIRE WORLD MAP simply to make it better.
- `1012790_10355 review=3 sentence=sentence_20`: Now the world map is split into sections, all of which have multiple entry points (already an improvement), interconnect to each other, and some of which connect directly to your base.
- `1012790_10355 review=3 sentence=sentence_23`: And even more than that, maps can change according to you mission to add even more, different varieties to you travels.
- `1029690_19693 review=4 sentence=sentence_10`: 地图会标识任务目标位置，主角位置和朝向也有了清晰的标记。
- `1040200_11371 review=5 sentence=sentence_13`: After map ends it shows you if you found all secrets or not.
- `1040200_11371 review=5 sentence=sentence_16`: No long runs around the map.
- `1040200_11371 review=5 sentence=sentence_17`: Maps are very compact, water stations everywhere, so you can really spend the whole hour on a level cleaning it, not running from point A to B with a bucket.
- `1041720_6700 review=7 sentence=sentence_8`: А за счет того, что он огромен - реально огромен, чесслово, его исследовать можно чуть ли не столь же долго, как Морровинд из 3-й ТЕС - тут нашлось место и позитивненьким лесным полянкам, и жутковатым пустошам.
- `1058830_3923 review=4 sentence=sentence_14`: That's not even mentioning the maps themselves which are mostly fun and enjoyable, if not occasionally a bit stamina-testing.

### Topic 18 (419 docs)

Top words: devs, developers, dev, developer, team, community, feedback, dev team, work, listen

- `1003590_7381 review=5 sentence=sentence_13`: Now the other game developers, they noticed this.
- `1013320_5961 review=3 sentence=sentence_12`: Oh look the devs responded.
- `1016800_13368 review=3 sentence=sentence_7`: But you can tell the devs poured their hearts out into this.
- `1016950_4196 review=5 sentence=sentence_12`: This means that the developers are only tasked with making it work and making it look good.
- `1020790_3591 review=5 sentence=sentence_6`: Devido ao trabalho preguiçoso por parte dos desenvolvedores.
- `1029780_17642 review=4 sentence=sentence_3`: So far the devs also seem to be passionate.
- `1029780_17642 review=4 sentence=sentence_5`: As for the devs, right now more than anything else you need to integrate mod and workshop support.
- `1034140_19644 review=3 sentence=sentence_44`: 하지만 유일하게 염려되는게 있다면 개발진들의 취향입니다..
- `1045720_3045 review=3 sentence=sentence_7`: Emeği geçen herkese ve Localsheep çeviri ekibine sonsuz teşekkürlerimi iletiyorum :>
- `1045720_3045 review=4 sentence=sentence_2`: Розробники Devespresso взяли усі круті елементи, вдосконалили їх і приправили ще крутішими.

### Topic 19 (418 docs)

Top words: controller, controls, mouse, keyboard, button, controle, xbox, buttons, press, wasd

- `1001270_4904 review=5 sentence=sentence_9`: - No custom keybind option. - The game does not lower its power/CPU/GPU/RAM usage or FPS when you tab out.
- `1001270_4904 review=5 sentence=sentence_16`: - Turning on and off the Grid in Edit Mode (hit "G") will also open/collapse the pinned recipe in the bottom left (also hit "G").- If you have open the pinned recipe window in the bottom left when you open the tablet, it will stay open and appear on top of the tablet UI.
- `1016790_1893 review=3 sentence=sentence_13`: Game is clearly designed for a controller rather than a keyboard and mouse•
- `1022310_1594 review=4 sentence=sentence_39`: There aren't even labels/tooltips on some of the buttons.
- `1029550_11568 review=6 sentence=sentence_3`: I wasn't a fan of the control scheme, and iOS controller support wasn't available yet.
- `1029550_11568 review=6 sentence=sentence_7`: The Pros:- Driving is fun on an Xbox One controller.
- `1029550_11568 review=6 sentence=sentence_8`: Takes some tweaking (controller smoothness etc), but once you learn it, it's good.
- `1041720_6700 review=3 sentence=sentence_3`: Well the fact is you can change your keybinds in this game, and I'm pretty sure you could do it in the old game too.
- `1041720_6700 review=3 sentence=sentence_4`: It's in this really mysterious and hidden location in the options menu called...Controls.
- `1041720_6700 review=3 sentence=sentence_6`: Update: Beware that as of right now, mouse button binding doesn't work.

### Topic 20 (412 docs)

Top words: review, reviews, negative, negative review, write, write review, update review, read, original review, negative reviews

- `1016800_13368 review=3 sentence=sentence_3`: (Which might be the reason for the negative reviews.
- `1029550_11568 review=5 sentence=sentence_1`: Отзывы никто не читает, поэтому напишу рецепт блинов.-
- `1038250_6432 review=4 sentence=sentence_21`: Enfim, vou tentar encerrar por aqui as críticas.
- `1043810_6382 review=3 sentence=sentence_37`: Will do more testing, will probably update review as that happens.
- `1051690_3450 review=3 sentence=sentence_2`: I recommend you read the full review for the full detail.
- `1051690_3450 review=3 sentence=sentence_5`: I think I can write this review pretty confidently lol.
- `1051690_3450 review=3 sentence=sentence_35`: ...Okay, this section of the review sounds like a snobby game journalist wrote it, but I stand by my words!
- `105600_54613 review=6 sentence=sentence_5`: If you're still reading this review, what are you doing?
- `1056640_23530 review=3 sentence=sentence_34`: Feel free to throw anymore QUICK advice and I'll add it to the review!
- `1056640_23530 review=3 sentence=sentence_39`: Lastly, please give this review an award of any kind if you found it useful!

### Topic 21 (402 docs)

Top words: weapons, weapon, ammo, guns, gun, rifle, shotgun, rifles, melee, shotguns

- `1000360_6765 review=3 sentence=sentence_5`: Your weapon has a hidden, literal weight stat to it in grams (longsword being the heaviest, rapier the lightest).
- `1000410_2404 review=3 sentence=sentence_8`: -Weapons for the most part are a very traditional roster, but they have some unique designs and each has an altfire, though some of those altfires are much less useful than others.
- `1000410_2404 review=3 sentence=sentence_9`: Ammo distribution for each weapon feels all over the place, however.
- `1000410_2404 review=3 sentence=sentence_44`: Enemies and weapons are still being developed and tested.
- `1012790_10355 review=4 sentence=sentence_5`: >try to pull out my shotgun but he's getting too close
- `1012790_10355 review=5 sentence=sentence_1`: >have enough money to buy ppsh>buy ppsh>yay>go to buy ammo>only 150 a mag>buy 5>go into the radius (tm)>encounter a wooden post with supplies >encounter headcrab looking mfs>take out ppsh>doesnt shoot>♥♥♥♥
- `1012790_10355 review=5 sentence=sentence_3`: nothing works.>take out mag, look inside said mag>no bullets
- `1012790_10355 review=5 sentence=sentence_4`: I had bought 5 CLIPS with no bullets
- `1016790_1893 review=3 sentence=sentence_8`: Weapons are standard, nothing unique or exciting about them•
- `1016790_1893 review=3 sentence=sentence_30`: Even when it comes to weapons, there was always a specific set I was on the lookout for because it was just better than the others.

### Topic 22 (395 docs)

Top words: review, recommend, recommend game, reviews, negative, negative reviews, game, review game, positive, negative review

- `1016950_4196 review=5 sentence=sentence_2`: As a major Blood Bowl fan, who has invested hundreds of hours in Blood Bowl 2 and has actively promoted the game, I can and will NOT recommend this cesspool of disappointment.
- `1017900_1112 review=4 sentence=sentence_1`: Impossible en l'état de recommander ce jeu.
- `1049410_23830 review=6 sentence=sentence_6`: Some people say it's invalid to directly compare games in criticism because the reader isn't guaranteed to have played those.
- `1056640_23530 review=3 sentence=sentence_44`: If you want me to add any advice to this review, feel free to leave a comment, because since I stopped playing, I don't know how to update this any further.
- `1056960_2523 review=3 sentence=sentence_48`: This game offers way more than reviews make it seem.
- `1057750_6138 review=4 sentence=sentence_5`: The plot, however, is so terrible that I cannot in good conscience recommend this game.
- `1058650_7389 review=8 sentence=sentence_1`: - Public conseillé :☐ Je ne conseillerai pas ce jeu même à mon pire ennemi☐
- `1062520_16450 review=4 sentence=sentence_8`: This is the first game I've ever felt compelled to leave a review for because I heard it was a solo project!!
- `1063420_3769 review=5 sentence=sentence_13`: But, due to the mismatch between the marketing and the actual game, I can't recommend it.
- `1063660_11863 review=5 sentence=sentence_15`: I do still overall recommend the game - but only for those who are okay with a shorter experience and mostly want to play for the story telling and rich atmosphere.

### Topic 23 (389 docs)

Top words: quake, house flipper, flipper, quake quake, house, champions, overcooked, nintendo, h1, spoiler

- `1030210_11120 review=5 sentence=sentence_27`: Самое интересно началось уже после того как я начал играть непосредственно в "игру"."Игра" буквально надменный, нахальный плевок в лицо всем фанатам этой серии.
- `1030210_11120 review=5 sentence=sentence_29`: Как бы описать геймплей...
- `1043260_3322 review=3 sentence=sentence_9`: Графика прикольная и игровой процесс затягивает.
- `1045720_3045 review=4 sentence=sentence_8`: Ці команди не набридають, як у багатьох ігор, а доречно лягають в канву гри й додають напругу.
- `105600_54613 review=8 sentence=sentence_2`: Игра улёт , многие критикуют за то что это 2д "кубики" и сравнивают с майнкрафтом.
- `105600_54613 review=8 sentence=sentence_9`: Игра может сразу не понравиться , но за её не замысловатым видом скрывается большой потенциал и большая вселенная где всё ограничевается только твоей фантазией .
- `1069690_1296 review=4 sentence=sentence_8`: Многие негативные отзывы связаны с тем, что эту игру очень сложно правильно позиционировать в глазах потенциального покупателя.
- `1069690_1296 review=4 sentence=sentence_9`: Если вы пришли за игрой, где можно чудовищами распиливать беззащитных людей на части, или ожидаете какой-то аналог Iratus - эта игра вам этого не даст.
- `1072040_3577 review=4 sentence=sentence_2`: Заставляет вспоминать, как первоклассником в 1994 году смотрел за играющим в Panzer General отцом, восхищаясь умениями и навыками, а сам мог выиграть только Польшу.
- `1072040_3577 review=4 sentence=sentence_10`: Так вот, вышла похожая игра.

### Topic 24 (385 docs)

Top words: roguelike, rogue, doom, roguelite, rpg, souls, diablo, skyrim, dungeon, like

- `1016790_1893 review=3 sentence=sentence_35`: Compare this to other rogue-like games like and you’ll find that you can unlock a bunch of new weapons within the first few hours of that game and it feels satisfying and exciting to unlock them.
- `1016790_1893 review=3 sentence=sentence_37`: Difficulty is another aspect of a rogue-like that’s important to get right and unfortunately doesn’t get this right either.
- `1016790_1893 review=3 sentence=sentence_49`: There are definitely much better rogue-likes out there than this one, I wouldn’t recommend getting this one unless you're desperate for a new rogue-like.
- `1016800_13368 review=3 sentence=sentence_4`: Many people expected this to be like S.T.A.L.K.E.R)
- `1016800_13368 review=4 sentence=sentence_6`: Some key points about gameplay:RPG Elements-
- `1036890_4908 review=3 sentence=sentence_17`: I complained that SW2 felt too much like the devs wanted to recreate borderlands, but here the comparison is DOOM, which I'm not going to complain about because everyone else has already brought this up.
- `1036890_4908 review=4 sentence=sentence_1`: It has great graphics, good performance , and the gameplay is fun and fast paced, kinda like Doom.
- `1037020_2760 review=3 sentence=sentence_40`: 最后还有一点，《ScourgeBringer》作为一款Rogue，尽管有着随机生成的关卡、敌人以及奖励道具，但实质玩起来和线性推关完全没区别。
- `1037020_2760 review=4 sentence=sentence_3`: Rougelike 不可或缺的游戏核心在于「带有随机化的新内容」，虽然重复，但不单调，这意味着你每次的游戏过程都会有较为明显的差异性，可能是装备、可能是技能……
- `1040420_2914 review=3 sentence=sentence_1`: Yes, I'd recommend Dreamscaper to any fan of the roguelite genre.

### Topic 25 (378 docs)

Top words: simulator, stardew, sims, stardew valley, sim, valley, walking, farming, walking simulator, simulation

- `1001270_4904 review=4 sentence=sentence_9`: Daha ileri zamanlarda kesme mekaniği cooking simulatordeki gibi olabilir.
- `1001270_4904 review=4 sentence=sentence_11`: Kebap sim yerine direk restoran simulator olarak değişirse falan baya güzel olur.
- `1028310_3994 review=3 sentence=sentence_1`: I once left a bad review of Cultist Simulator because it made me feel very much like Brooks in Shawshank Redemption:
- `1028310_3994 review=4 sentence=sentence_7`: It has a nice amount of that same CULTIST SIMULATOR flavor, but with less pressure, less stress.
- `1028310_3994 review=4 sentence=sentence_17`: In Cultist Simulator that was more 'ok', because you would read a book and it would disappear and give you some knowledge and you'd be good to go.
- `1040200_11371 review=5 sentence=sentence_2`: I love cleaning simulators and I have tried them all, but this might be the best.
- `1040200_11371 review=5 sentence=sentence_12`: In a cleaning simulator.
- `1062520_16450 review=3 sentence=sentence_2`: It brings a lot of the feel of Animal Crossing, but give you more options with what to do with your time.
- `1062520_16450 review=4 sentence=sentence_1`: The perfect farming game if you ever found Animal Crossing too slow and Stardew Valley too fast + broad!
- `1062520_16450 review=4 sentence=sentence_3`: I struggled a lot with enjoying Animal Crossing because there wasn't enough to do in a day and Stardew Valley made me feel like I was choking on time.

### Topic 26 (376 docs)

Top words: horror, horror game, resident evil, resident, evil, survival horror, scary, survival, scares, horror games

- `1035120_4085 review=3 sentence=sentence_4`: There's this whole otherworldly horror thing and the porn around it feels uncanny and cringe, and yet enticing at the same time - it's a hard trick to pull, especially today when everyone is accustomed to porn.
- `1035120_4085 review=3 sentence=sentence_15`: Pros:- horror erotica done right- atmosphere, visuals and audio- detail of the world- story
- `1035120_4085 review=4 sentence=sentence_6`: A Lovecraft/Giger inspired horror themed mystery story with strong adult themes, some times uncomfortable ones.
- `1035120_4085 review=7 sentence=sentence_10`: Es un juego del género Survival Horror con toques de erotismo.
- `1035120_4085 review=7 sentence=sentence_11`: Al menos es más Survival Horror de lo que son los Resident Evil 4, 5 y 6.
- `1051690_3450 review=3 sentence=sentence_32`: Nightmare Reaper does not disappoint.
- `1063660_11863 review=4 sentence=sentence_7`: In terms of gameplay, there's a much bigger emphasis on the "survival" aspect of the horror, something I felt was lacking in and is a welcome change.
- `1063660_11863 review=5 sentence=sentence_14`: Whereas I think the first Bendy was more revolutionary in the horror genre, maybe I was just expecting more out of the developers due to the time taken to make BATDR as well as it being a sequel.
- `1069160_2309 review=4 sentence=sentence_13`: well-written, horrifying, electrifying, and just fun.
- `1069690_1296 review=3 sentence=sentence_12`: The Terror effects and drawbacks are also a good and interesting part of the gameplay.

### Topic 27 (369 docs)

Top words: translation, english, language, chinese, traduo, translate, bersetzung, bersetzt, idioma, localization

- `1030830_26263 review=4 sentence=sentence_2`: открываем с помощью блокнота, находим строку в самом начале <Language>english</Language> и меняем на russian, сохраняем и заходим в игру.
- `1040200_11371 review=3 sentence=sentence_1`: 안녕하세요, 이번에 크라임 씬 클리너에서 인게임 한국어 텍스트와 공지사항을 번역하고 있는 사차지라고 합니다
- `1040200_11371 review=3 sentence=sentence_2`: 제가 개발자 분에게 '내가 따로 번역 수정사항을 지정할 수 있는 게 없는가?' 해서 어쩌다보니 진짜 개발자 분들이 쓰시는 번역 사이트의 권한을 얻고 굳이 필요없는 전체 텍스트 수정과 번역을 진행하게 됐습니다
- `1040200_11371 review=3 sentence=sentence_3`: 하하참고로 다른 한국인 분들 없이 오로지 저 혼자 텍스트를 번역했기 때문에 어색한 부분이 있을 수 있습니다
- `1040200_11371 review=3 sentence=sentence_5`: 게임 정식 출시가 되지 않은 몇 달 전부터 현재까지 쭉 모든 텍스트들을 검토하고 수정했는데요, 한국 유저 분들을 생각하며 번역에 온 힘을 쏟았고 최종 퀄리티를 여러분들에게 선보이니 제가 다 기쁜 것 같습니다
- `1040200_11371 review=3 sentence=sentence_8`: 사전에 이미 몇 번이고 전체 텍스트들을 검토하고, 일부는 수정한 상태지만 여전히 어색한 부분이나 게임과 맞지 않은 텍스트들이 존재합니다
- `1040200_11371 review=3 sentence=sentence_9`: 그럴 경우 저에게 따로 전달 해주시거나, 공식 디스코드를 통해 개발자 분에게 전달을 주시면 번역에 큰 도움이 될 것 같습니다
- `1040200_11371 review=3 sentence=sentence_11`: 저는 이미 사전에 번역 검토를 하느라 모든 챕터를 클리어하고 온 상태인데 네..
- `1040200_11371 review=3 sentence=sentence_18`: 자발적으로 지원해서 번역을 담당하게 된 거예요!
- `1040200_11371 review=3 sentence=sentence_20`: 영어에 대한 지식이 낮고 당연히 전문적인 지식을 갖추지 않고 번역에 뛰어들었습니다

### Topic 28 (361 docs)

Top words: puzzles, puzzle, puzzle game, solving, solve, puzzle games, easy, picross, puzzles game, riddles

- `1003590_7381 review=5 sentence=sentence_1`: Tetris is the king of the puzzle games.
- `1022980_1451 review=5 sentence=sentence_4`: Lots of very hard puzzles and if you make a single mistake there is a 95% chance you will die.
- `1035120_4085 review=7 sentence=sentence_12`: Similar al 7; tendremos que resolver puzzles, buscar llaves y el combate es uno de los últimos recursos ya que te pueden hacer pelota muy fácil y algunos enemigos es mejor ni tocarlos.
- `1042490_4762 review=3 sentence=sentence_14`: I really enjoyed this part of the game as it made it feel like more than just a puzzle game loosely tied together.
- `1042490_4762 review=3 sentence=sentence_20`: Puzzles were imaginative but quite difficult I thought.
- `1042490_4762 review=3 sentence=sentence_22`: One of these puzzles is broken and the other was slightly misleading.
- `1042490_4762 review=3 sentence=sentence_23`: The puzzles were cleverly entwined around the story, which is mostly Polynesian based, and added to the immersion and feeling that the puzzle was part of finding Harry rather than a disconnected puzzle for the sake of it.
- `1042490_4762 review=3 sentence=sentence_27`: I think most people will struggle with these puzzles.
- `1042490_4762 review=3 sentence=sentence_50`: The puzzles are varied and interesting and can be quite challenging.
- `1043810_6382 review=3 sentence=sentence_1`: This is one of the most well-crafted turn-based/puzzle games to come out in recent memory.

### Topic 29 (348 docs)

Top words: story, ending, plot, interesting, story story, end, storytelling, characters, emotional, narrative

- `1016800_13368 review=3 sentence=sentence_27`: (With a Fallout-esque tapestry ending and everything)
- `1040070_2071 review=4 sentence=sentence_15`: In addition, there's a very cute story that slowly unravels.
- `1040070_2071 review=4 sentence=sentence_17`: What's also great is that the story IS worth investing in and journeying all across the world for.
- `1040200_11371 review=5 sentence=sentence_18`: An overarching story that is actually good.
- `1042490_4762 review=3 sentence=sentence_8`: There is even a short, but informative, seven-part prequel story in the news section to wet your appetite and introduce you to the two main characters.
- `1057750_6138 review=4 sentence=sentence_27`: Not to be petty, but it's also straight up bad story-telling.
- `1063660_11863 review=5 sentence=sentence_5`: Great story tellingCons:-
- `1069530_1521 review=5 sentence=sentence_15`: The story is simple but it's told in such an obtuse and frustrating way you simply don't care.
- `1069530_1521 review=5 sentence=sentence_22`: The entire story needs to be re-written to make it less frustrating to read.
- `1082430_15462 review=4 sentence=sentence_5`: 5- Cry like a monster at the end.

### Topic 30 (310 docs)

Top words: enemies, enemy, attack, dodge, attacks, hit, cover, block, enemy types, enemy attacks

- `1000360_6765 review=3 sentence=sentence_8`: You need to think very carefully about your approach to your opponent, and you need to pay attention to the vulnerabilities in your opponent's play to determine the correct angle for attack.
- `1000360_6765 review=3 sentence=sentence_9`: Alternatively, if you choose the wrong angle, you WILL be parry riposted and murdered in cold blood.
- `1000360_6765 review=3 sentence=sentence_19`: This really drives home the point of choosing your approach intelligently, as opposed to relying on muscle memory and the same attacks every time.
- `1000360_6765 review=4 sentence=sentence_5`: Your defense depends on your stance, as it should, and finding out what every weapon, stance and angle does and using that knowledge to outsmart your opponent on the fly is an absolute delight.
- `1000410_2404 review=3 sentence=sentence_16`: Enemy counts sometimes rival
- `1000410_2404 review=3 sentence=sentence_19`: -Enemy variety is too lacking for how large each level is.
- `1000410_2404 review=3 sentence=sentence_22`: -While enemies are visually distinct and easy to recognize at a distance, most fulfill the same exact purpose.
- `1012790_10355 review=3 sentence=sentence_14`: You almost fight exclusively like 2 enemies ever, which is a shame cause there's a whole lot of them to use and fighting against some of them just becomes a hassle in environments that make fighting either way too close or way too far (staying vague so I don't spoil enemy discovery cause that's a lot of the fun).
- `1016790_1893 review=3 sentence=sentence_16`: Sometimes when vaulting over cover you might go through it instead of over it•
- `1016790_1893 review=3 sentence=sentence_18`: However, this falls apart when the introduction of melee enemies comes into play, these enemies will come running at you and if you're taking cover you are practically forced to retreat out of the room whilst shooting at them.

### Topic 31 (303 docs)

Top words: animals, cat, dogs, animal, pet, dog, cats, horse, pets, animais

- `1009560_6684 review=4 sentence=sentence_13`: Others couldn't get past your new tutorials to just enjoy their cute new pet pigs, or little puppies with big wagging tails.
- `1028310_3994 review=4 sentence=sentence_45`: I can't even decorate or buy fish for my cat!), and going to Brancrug is a huge chore.
- `1062110_1737 review=3 sentence=sentence_45`: The dogs you save cannot be hurt, let loose the dogs of D'aaww/
- `1075740_2952 review=4 sentence=sentence_16`: 希望官方能看到吧1.希望各个动物能添加一个固有特征
- `1112890_4627 review=3 sentence=sentence_2`: The premise of Calico is "what if Animal Crossing, but you run a cat cafe?"
- `1112890_4627 review=3 sentence=sentence_6`: You can pet every single animal in this game, and there is exactly one animation for all of them.
- `1112890_4627 review=3 sentence=sentence_7`: This means that you pet a cat the exact same way that you pet a dog, horse, pig, deer, raccoon, grizzly bear…
- `1112890_4627 review=3 sentence=sentence_8`: So you’re not really petting a crow, you’re petting a loud black cat with wings.
- `1112890_4627 review=3 sentence=sentence_12`: You know that part where I said that this is Animal Crossing but you run a Cat Cafe?
- `1112890_4627 review=3 sentence=sentence_20`: Animals behave like this as well.

### Topic 32 (292 docs)

Top words: mod, chinese, bug, milk1, tiny bunny, deeply, swipe, chinese version, adv, translated

- `1022980_1451 review=4 sentence=sentence_25`: 本作的开发商和发行商之前似乎有表示在正式版时会有官方中文。
- `1022980_1451 review=4 sentence=sentence_28`: 我个人会尝试在EA阶段就制作中文汉化补丁方便大家使用。
- `1034140_19644 review=3 sentence=sentence_1`: Subverse 0.1.3 얼리억세스 버전의 준 한글화를 완료하였습니다.
- `1049890_11254 review=3 sentence=sentence_54`: 可以打汉化MOD或补丁√ 自带汉化！☐ 自带汉化和中文配音！！---{价格}---☐ 免费游玩!☐
- `1075740_2952 review=4 sentence=sentence_14`: 7.简中有些地方翻译有问题，像这种卡牌类游戏要是翻译不精准玩的时候非常容易暴毙，希望能赶紧修正
- `1083880_2779 review=5 sentence=sentence_1`: ※2024/10/10: 從遊戲發行到現在歷經將近五年的時間，官方終於提供簡體中文語言。
- `1083880_2779 review=5 sentence=sentence_3`: 儘管對於他們的辛勞努力我們應該給予感謝才對，不過中文翻譯的品質我只能說「有就好」，看不懂的對話就切成英文來看，可以的話用原文來玩比較原汁原味，英文單字沒有難到完全看不懂的程度。
- `1100410_2189 review=3 sentence=sentence_56`: 打上民间自制的高清补丁，完胜这个重制版，打上“目标巴黎”mod，体验全新内容。
- `1107790_3157 review=5 sentence=sentence_11`: 还增加了亲密度影响剧情的设定，而且全文汉化对中国玩家也很友好，感觉游戏制作组还是很用心的。
- `1113570_1295 review=3 sentence=sentence_3`: 算了，不重要，看到评测的人只要知道这篇评测发布的前一晚，简中汉化补丁终于上线就够了。

### Topic 33 (292 docs)

Top words: early access, access, early, game early, access game, release, alpha, released, access early, access release

- `1000360_6765 review=3 sentence=sentence_24`: Obviously as a "pre-alpha" early access release, the game isn't incredibly polished and you do end up seeing some weird, jank interactions from time to time.
- `1000410_2404 review=3 sentence=sentence_32`: Wrath currently has the first episode available in early access: 5 levels and a hub world.
- `1000410_2404 review=3 sentence=sentence_33`: With its original launch window, this was perfectly acceptable; a shareware-like build with the full release coming relatively soon.
- `1000410_2404 review=3 sentence=sentence_34`: The original release window was set for early 2021 (early access began in late 2019).
- `1000410_2404 review=3 sentence=sentence_42`: The first bi-weekly updates they release showed that the game was barely developed past what is in the early access build.
- `1000410_2404 review=3 sentence=sentence_53`: I mean, based on the posts we were seeing, the game didn't have a chance of hitting that release window, but seriously?
- `1001270_4904 review=5 sentence=sentence_6`: This one is unique in how you be reminded of how it's early access almost the entire time you play the game.
- `1016790_1893 review=3 sentence=sentence_48`: It feels like an Early Access game with its lack of polish and general balancing issues.
- `1022980_1451 review=5 sentence=sentence_1`: WARNING1: This game is extremely early access, expect lots of bugs and buy only if you want to support and have faith in the developer
- `1029780_17642 review=6 sentence=sentence_1`: No soy de las personas que hacen reseñas en Steam, pero con este juego me es imposible no reconocer el enorme y muy pulido trabajo que han hecho con Going Medieval incluso estando en Early Access, el nivel de organización que demuestra su equipo de desarrollo se puede apreciar sencillamente al ver la interfaz del juego.

### Topic 34 (265 docs)

Top words: houses, build, building, buildings, village, furniture, house, bricks, placed, base

- `1001270_4904 review=5 sentence=sentence_22`: This one hurts most for people who want to decorate in unique ways: the wallpaper and the floor (at least in the first restaurant) do not line up.-
- `1017900_1112 review=4 sentence=sentence_9`: - Impossible de cumuler les ordres de construction
- `1040070_2071 review=3 sentence=sentence_15`: So add a half-thumb up if you just like building yourself up without having to worry about anything at all.
- `1060230_1945 review=3 sentence=sentence_7`: You are not limited to pre-designed building constructs.
- `1060230_1945 review=3 sentence=sentence_8`: Instead, you get to design your buildings wall-by-wall, roof-by-roof.
- `1060230_1945 review=4 sentence=sentence_12`: By that I mean it could be refined with a bit more code, but the villagers don’t do anything random or get stuck.
- `1073910_1574 review=3 sentence=sentence_9`: Then, if you do, you find that you don't have enough space in your warehouses for the supplies so you need to build a settlement worth of warehouses.
- `1104330_2060 review=4 sentence=sentence_8`: 이를 이용해 상인이나 동맹고블린을 원하는 위치에 데려올 수 있다.
- `1104330_2060 review=4 sentence=sentence_12`: 2층집을 지을때는 바닥건설에서 R을 눌러 건설하자.
- `1104330_2060 review=4 sentence=sentence_14`: 난로가 1층에 있으면 2층건설을 방해할 수 있다.)

### Topic 35 (262 docs)

Top words: enjoyed, fun, enjoying, really, like, really enjoyed, enjoy, liked, enjoyed time, did

- `1012790_10355 review=6 sentence=sentence_15`: I am having a great time.
- `1015500_833 review=4 sentence=sentence_4`: Per il resto l'ho trovato giocabile e divertente.
- `1015500_833 review=9 sentence=sentence_17`: Hallelujah! - it worked and I got an enjoyable match out of it.
- `1030830_26263 review=6 sentence=sentence_2`: My friends can attest to this.
- `1035120_4085 review=4 sentence=sentence_7`: I like that sort of thing.
- `1041920_1898 review=4 sentence=sentence_11`: In that moment, I understood I was not preordained to forever be the quiet person in the corner, the one whose inexpressive attire blended in with the world around them.
- `1049410_23830 review=4 sentence=sentence_7`: It resonated with me so much, at that point in my life, that I cried.
- `1051690_3450 review=3 sentence=sentence_8`: Even now, it remains just as enjoyably addictive.
- `1065310_9049 review=4 sentence=sentence_5`: It was a lot of fun though!!!
- `1089980_30650 review=4 sentence=sentence_4`: You're going to love it.

### Topic 36 (260 docs)

Top words: mods, mod, modding, daggerfall, modding community, dot, daggerfall unity, community, unity, modded

- `1022980_1451 review=4 sentence=sentence_27`: 不过本作相比使用flash开发的前作，对Mod和其他语言的支持会好很多。
- `1029780_17642 review=4 sentence=sentence_6`: seriously drop every other thing you're doing and implement extensive mod tools.
- `105600_54613 review=8 sentence=sentence_5`: Но смотря на те возможности которые дает игра и площадка модов , это мелочь.
- `105600_54613 review=9 sentence=sentence_4`: Even if you've beaten the game 50 times over, there is still new stuff to do, and if you really get bored with the game, just check out tmodloader.
- `105600_54613 review=10 sentence=sentence_3`: With the amount of content in the game, and with the fact tModLoader is available and you can get mods like Calamity, Thorium, Terraria Overhaul and other mods that expand the game further, you are not gonna lose interest in the game for a long time.
- `107410_23399 review=7 sentence=sentence_1`: Mods make this game what it really is.
- `107410_23399 review=10 sentence=sentence_5`: And once you have tried the game, why not jump down the rabbit hole of mods, public servers and milsim.
- `107410_23399 review=10 sentence=sentence_7`: Even though the game is set in the future, there are mods that add all manner of contemporary military equipment from World War 1 up to modern day (including things like destroyers, submarines and even spy satellites).
- `1101190_4659 review=3 sentence=sentence_3`: All those CDT weapons, accessories, and pets were modeled and textured for this community, fueled by my passion for the game.
- `1148590_7255 review=5 sentence=sentence_4`: Find Doom 64 in your library list. Right-click and hover over the "Manage" tab, it should expand.

### Topic 37 (251 docs)

Top words: characters, character, main, memorable, personajes, protagonists, endearing, suffering builds, characters characters, builds character

- `1030830_26263 review=5 sentence=sentence_26`: It really makes the characters less expressive and more robotic.
- `1041920_1898 review=4 sentence=sentence_24`: Her characterization displays a rare sympathy.
- `1045720_3045 review=4 sentence=sentence_13`: Цікаві, зі своїми характерами.
- `1056960_2523 review=3 sentence=sentence_44`: Especially from people that had the main problem with - having female protagonist, calling them cringe or whatnot.
- `1069530_1521 review=5 sentence=sentence_11`: Every single screen in every area oozes with personality.
- `1084640_2165 review=3 sentence=sentence_36`: Химия между персонажами так и бурлит.
- `1087760_2041 review=3 sentence=sentence_5`: On one side you have a run-of-the-mill, stereotyped sci-fi trope recycled a million times before, one the other a main duo that never really stands out for neither personality nor charisma.
- `1087760_2041 review=3 sentence=sentence_6`: The other characters found along the adventure are sideshows at best, even more unremarkable, with forgettable and uninspired dialogues.
- `1088850_4860 review=5 sentence=sentence_5`: Its impressive the amount of dialogue they made in order to feel the constant banter between the characters.
- `1088850_4860 review=6 sentence=sentence_9`: Each of the Characters are each a healthy mix between the original comic characters and their James Gunn movie counterparts.

### Topic 38 (247 docs)

Top words: ai, pathing, bots, ia, ai ai, pathfinding, ai just, ai doesnt, la ia, ai really

- `1015500_833 review=5 sentence=sentence_2`: AI kick out after I've spent 20 minutes squashing them.
- `1015500_833 review=5 sentence=sentence_3`: I let an AI 1v1 match play for an hour without a winner, so I assume the AI are just bugged to be unbeatable.
- `1016790_1893 review=3 sentence=sentence_22`: The pathing for the AI can be pretty bad as well sometimes if you go into another room with an enemy following you they just try to beeline to you and consequently just walk into the corner of the room they’re in.
- `1017900_1112 review=5 sentence=sentence_6`: Cons: - Pathfinding...
- `1038250_6432 review=4 sentence=sentence_9`: Negativos...:-IA muito fraca, os bots são muito ruins.
- `1054490_9630 review=5 sentence=sentence_4`: Yes, you can finish against local AI, but you cannot continue online.
- `1060230_1945 review=4 sentence=sentence_11`: Sapient AI is simple but bug free and predictable.
- `1060230_1945 review=4 sentence=sentence_13`: For example, an idle villager will “steal” the job of a better candidate who is a lot closer, so the AI is inefficient.
- `1066890_9246 review=3 sentence=sentence_34`: Selling point #6: The AI are continually improved, continually enhanced, and they have gotten to a point where, if you manage to find a combination where they are on your pace (which is by far not as hard as it used to be), they are very likely to provide a fun experience.
- `1069650_1910 review=3 sentence=sentence_8`: 특히 인공지능이 많이 부족해서 일일이 손으로 만져줘야 하는 점이 큰 단점입니다, 정식 출시나 할인을 기다리시는 것도 좋아요.

### Topic 39 (236 docs)

Top words: difficulty, hard, difficulties, difficult, difficulty settings, harder, levels, challenging, hard mode, challenge

- `1000030_2098 review=3 sentence=sentence_37`: The 'choose your own difficulty' manner of the CSD series' menu configurations means you could probably replicate similar circumstances on another level if you really wanted - I can't speak for how missing out on this small subset of content in an official capacity might feel for those unable to access it.
- `1022980_1451 review=5 sentence=sentence_2`: WARNING2: This game is very hard, unforgiving and with very little hand holding.
- `1043260_3322 review=7 sentence=sentence_6`: Oyundaki arena sistemi 3 zorluk seviyesinden oluşuyor.
- `1043810_6382 review=3 sentence=sentence_24`: If you play through the game on Normal and then expect a challenge for Hard, you will be very disappointed.
- `1058830_3923 review=3 sentence=sentence_14`: 从而真正的掌握了游戏的玩法，逐渐可以调高难度。
- `1058830_3923 review=4 sentence=sentence_9`: For new players to the rhythm game genre, however, difficulty is gradually eased and thoroughly marked, making the game a welcoming experience to new players as well.
- `1060230_1945 review=3 sentence=sentence_2`: How I was going to rate this game was difficult.
- `1062110_1737 review=3 sentence=sentence_41`: Mind you, this isn't a terrible thing to have available for players that are struggling through the game or just in one area, which in addition to its other difficulty options, make the game very accessible to players of all skill levels.
- `1065310_9049 review=6 sentence=sentence_12`: Players who are not experienced in hack n slash games will struggle in Evil West.
- `1084160_11858 review=3 sentence=sentence_17`: I would recommend most players play on the default difficulty with the "To the bitter End" turned on.

### Topic 40 (235 docs)

Top words: good game, great game, good, amazing, game, great, game amazing, game great, game game, amazing game

- `1016800_13368 review=3 sentence=sentence_5`: And the game is actually REALLY good on it's own.
- `1016800_13368 review=4 sentence=sentence_23`: All-in-all this is a pretty great game.
- `1017900_1112 review=3 sentence=sentence_17`: everything else about the game is great.
- `1030840_80583 review=5 sentence=sentence_7`: Now it's a great game :)
- `1036890_4908 review=3 sentence=sentence_3`: After playing this game however, SW2 feels like a pretty great game in retrospect.
- `1040070_2071 review=4 sentence=sentence_1`: Unironically an outstanding game.
- `1040070_2071 review=4 sentence=sentence_8`: But this game is fantastic.
- `1055540_1264 review=7 sentence=sentence_1`: An absolutely beautiful game.
- `1061910_14713 review=5 sentence=sentence_2`: Beyond that complaint, this game is AMAZING.
- `1063730_82978 review=3 sentence=sentence_6`: Gerçekten çok zevkli bir oyun.

### Topic 41 (218 docs)

Top words: really really, recommend, recommend game, highly, really, highly recommend, game recommend, game highly, game, definitely recommend

- `1000360_6765 review=4 sentence=sentence_7`: All in all, I highly recommend this game.
- `1012790_10355 review=3 sentence=sentence_29`: Needless to say, I can highly recommend the game in its current form, hell I could have given it a moderately strong recommendation in the 1.0, but now I can even promise, it will get even better than it is now, and I already love it.
- `105600_54613 review=10 sentence=sentence_1`: Definitely a must play for anyone who enjoys games like Minecraft.
- `1058650_7389 review=3 sentence=sentence_15`: Do I recommend this game?
- `1067540_2680 review=5 sentence=sentence_10`: Настоятельно рекомендую эту игру каждому любителю адвенчур.
- `1104330_2060 review=3 sentence=sentence_24`: 무조건 추천드리는 게임이고 혹시 게임하시면서 궁금하시는거 있으시면 댓글로 적어주시면 하루에 한번정도는 보고 답할게요.
- `1126320_7150 review=3 sentence=sentence_39`: I recommend you play them!
- `1137460_10661 review=10 sentence=sentence_13`: Bu nedenle ilgili oyunu oynamanızı tavsiye ediyorum.
- `1139980_11466 review=4 sentence=sentence_23`: If they removed those limitations I would 100% recommend the game.---------------------------------------------------------------------------------------------UPDATE:
- `1147860_3675 review=4 sentence=sentence_11`: highly highly recommend if you find games as an artform engaging or interesting at all.

### Topic 42 (210 docs)

Top words: bugs, bug, updates, patch, patches, update, fixed, fix, minor, issues

- `1000360_6765 review=3 sentence=sentence_23`: ...unless a bug occurs.
- `1001270_4904 review=5 sentence=sentence_7`: Whether it be noticeable bugs, unpolished annoyances, or questionable ideas, you encounter them.
- `1022310_1594 review=4 sentence=sentence_12`: There are a lot of bugs.
- `1030830_26263 review=5 sentence=sentence_6`: Now for the issues that I've noticed that need to be fixed:1.
- `1030830_26263 review=5 sentence=sentence_18`: It would be nice to get a fix, but since this was an original issue, it could be a fundamental engine problem, so I won't be mad if it's not fixed.4.
- `1030830_26263 review=5 sentence=sentence_40`: In short, either do the community right for once and fix some bugs, or I and many others are out.
- `1054490_9630 review=4 sentence=sentence_20`: And it's not good that this has been a known bug for a while and the developers have not done anything to fix it and it doesn't look like they plan on doing anything to fix it.
- `1066890_9246 review=3 sentence=sentence_19`: Every 2-3 months, there's an update (and its usual hotfixes) to look forward to, and it's never a step back.
- `1069640_4384 review=5 sentence=sentence_6`: Es gab zwischendurch immer wieder Updates, Bugfixes und was auch immer sonst noch.
- `1069650_1910 review=4 sentence=sentence_3`: es sind noch nicht alle bugs beseitigt, mir ist aber bis auf wenige negative sachen nichts aufgefallen.

### Topic 43 (209 docs)

Top words: quests, quest, sidequests, main, complete, main story, story quest, fetch, story, main quest

- `1055540_1264 review=3 sentence=sentence_10`: And getting any side quests from NPCs felt nice cause they're all adorable :P
- `1062520_16450 review=3 sentence=sentence_30`: Quests don't check for if they're possible.
- `1062520_16450 review=5 sentence=sentence_8`: but you dont have a permanent place yet and cant finish or progress in town quests if you arent the host.
- `1062810_3287 review=5 sentence=sentence_8`: Massive cross-run quest chains, a persistent avatar, and annoyingly required story stuff are the scaffolding that support those runs: and they’re not always a positive inclusion.
- `1062810_3287 review=5 sentence=sentence_14`: Some are passive kill quests you’ll accumulate in the background of regular play, some are cool incentives to try a build you may not have otherwise, but the bad ones are corny RP quests and mandatory story deviations.
- `1062810_3287 review=5 sentence=sentence_15`: I once bottlenecked my progression because I didn’t realize a main story quest demanded that I play a particular class, go to a particular zone, then use an emoji.
- `1062810_3287 review=5 sentence=sentence_16`: Other quests force suboptimal runs by having you spend one of your path choices on talking to someone and getting a random boon rather than piloting your run the correct way: and the story just is not interesting enough to warrant forcing you away from a perfect build.
- `1069640_4384 review=5 sentence=sentence_27`: Individualisieren gibt’s nicht, außer die Kriegsbemalung, hier aber besser auf die Quest (genannt objective) warten.
- `1083880_2779 review=4 sentence=sentence_19`: コボルト姫の場合、キャンプ中に話が進行してLoveが高まると前提クエストが開始するので終わらせることでルートが始まる
- `1114220_3476 review=4 sentence=sentence_12`: Quests: I wouldn’t say the quests were the highlight of the game as yet, but they are sufficiently entertaining to hold your attention.

### Topic 44 (207 docs)

Top words: ship, ships, fishing, boat, fleet, boats, sea, minigame, underwater, atlantic

- `1022980_1451 review=3 sentence=sentence_13`: You'll start underwater on a mortgage, and you'll have to do whatever it takes to pay off your ship, and likely will be upgrading and retrofitting your ship to better suit your playstyle as you go.
- `1022980_1451 review=3 sentence=sentence_19`: Improving your skills, your ship, and with enough time and money, gradually mitigating and removing the scars that 0G living have left on your body.
- `1055540_1264 review=6 sentence=sentence_11`: I also would have liked to park my boat anywhere I please and not have it be warped back to the docking location once out of sight.
- `1063420_3769 review=3 sentence=sentence_4`: You didn't need that alloy to repair the 12 major hull breaches anyway!-
- `1063420_3769 review=3 sentence=sentence_6`: Accidentally take the ship's homunculus core out in the middle of a full scale firefight and watch the entire ship and all it's systems go offline, blame the gunner!-
- `1069160_2309 review=5 sentence=sentence_9`: Along the way, the ship's crew catches site of a monstrosity in the water and refuse to try to land at the town they were sailing to, leading to the game's first battle.
- `1069650_1910 review=4 sentence=sentence_8`: dieses system belohnt es nicht grössere schiffe zu haben und gut in einer mission abzuschneiden, weil man irgendwie immer wieder im nachteil ist.
- `1134100_3473 review=6 sentence=sentence_10`: With the assistance of tech upgrades, the boats can be equipped with a ballista and better sails so that you can get your pillage on faster.
- `1143810_3491 review=8 sentence=sentence_1`: It had Crafting systems, Father-ship customization of buildings, farming, crafting, Cooler ship gun placements, etc.
- `1150080_7309 review=3 sentence=sentence_31`: The game's 25 playable ships for the most part play the same, you have multiple types to choose from but in my experience ones, that matter is battleships and carriers.

### Topic 45 (206 docs)

Top words: recommend, recommended, highly, recommendation, highly recommended, recommend recommend, highly recommend, dont recommend, je, recommend highly

- `1015500_833 review=8 sentence=sentence_10`: Not recommended at all.
- `1028310_3994 review=4 sentence=sentence_48`: Despite my complaints, if you've read this far, you'll have noticed I gave this a recommendation anyways.
- `1043810_6382 review=3 sentence=sentence_39`: Experimental is not , but I recommend it.
- `105600_54613 review=7 sentence=sentence_18`: Рекомендую всем, кто ещё не опробовал.
- `105600_54613 review=8 sentence=sentence_8`: Вообщем рекомендую .
- `1057090_26450 review=5 sentence=sentence_16`: Highly recommended to everyone else.
- `1062520_16450 review=4 sentence=sentence_7`: I HIGHLY RECOMMEND :D
- `1066890_9246 review=3 sentence=sentence_45`: Very much recommended.
- `1069160_2309 review=4 sentence=sentence_14`: i cannot recommend it enough.
- `1075740_2952 review=5 sentence=sentence_4`: 親指を立てたものの、おすすめはできません。

### Topic 46 (202 docs)

Top words: pc, port, version, pc port, pc version, ps4, console, consoles, gen, ps2

- `1003590_7381 review=3 sentence=sentence_4`: Shame I didn't have a PS4.
- `1003590_7381 review=5 sentence=sentence_10`: Adaptations on stupid consoles.
- `1017900_1112 review=3 sentence=sentence_3`: the original is currently installed on my pc and i do play it from time to time.
- `1034140_19644 review=3 sentence=sentence_12`: 참고로 PC 사양을 크게 타지 않습니다.
- `1043810_6382 review=3 sentence=sentence_43`: I think if you are someone who will play this you will enjoy this than you would on PC.
- `1055540_1264 review=5 sentence=sentence_11`: Thanks for the excellent port.
- `1088710_8240 review=3 sentence=sentence_10`: Having originally bought and played through the game on PS3, with 30 FPS and some frame drops, I can't ever imagine going back to the earlier version.
- `1101190_4659 review=4 sentence=sentence_4`: This game will forever be shackled by the Switch conversion.
- `1101190_4659 review=4 sentence=sentence_8`: It is the first time I've heard of a company force downgrading an already released PC game to become a console game universally.
- `1105510_5640 review=4 sentence=sentence_2`: When i saw this coming out for PC with smooth 60fps 1080p it was an immediate purchase for me.

### Topic 47 (198 docs)

Top words: ai2lucy, ai2lucy got, city final, got problems, edition ai2lucy, final edition, scarlet maidenmaid, milfy city, maidenmaid mansion, problems scarlet

- `1097430_6948 review=3 sentence=sentence_10`: 淑女都市：Milfy City - Final Edition
- `1097430_6948 review=3 sentence=sentence_12`: AI诺娃-机娘育成方程式2露西惹了大麻烦：Lucy Got Problems
- `1097430_6948 review=3 sentence=sentence_13`: 绯红少女：Scarlet Maiden女仆洋馆：Maid Mansion
- `1097430_6948 review=3 sentence=sentence_18`: 暮绘朝花：Come Inside_又名记忆深处启示录/天欲启示录-：Apocalust
- `1097430_6948 review=3 sentence=sentence_24`: 精灵之妊 -用怀孕征服所有傲慢的精灵与经纪人恋爱是绝对禁止2圣石少女篇：OVER‧DeviL恶魔调酒师九点开张快捷情趣酒店：Quickie:A Love Hotel Story
- `1097430_6948 review=3 sentence=sentence_26`: 诱人的姐姐欲望岛：Lust Island[18+]大学女生的性生活：College Sex Party——————————————————————魔法家族
- `1097430_6948 review=3 sentence=sentence_37`: 我的性感表姐2：My Cute Roommate 2卧底之恋： Undercover Love
- `1097430_6948 review=3 sentence=sentence_38`: 幸运人生：Lucky Life性感女孩深入交流：Unknown Prodigy
- `1097430_6948 review=3 sentence=sentence_55`: 悲伤的天堂：Sorrow不雅的欲望：Indecent Desires大学琐事-学院情缘：University of Problems
- `1144400_27470 review=3 sentence=sentence_10`: 淑女都市：Milfy City - Final Edition

### Topic 48 (198 docs)

Top words: discord, banned, ban, community, bans, forums, discord server, permanently, server, post

- `1013320_5961 review=3 sentence=sentence_24`: No I don't want to join the discord.
- `1056640_23530 review=3 sentence=sentence_31`: So please go look it up on the PSO2 Subreddit, or something like that.[EPISODE 6 UPDATE]:
- `1058830_3923 review=4 sentence=sentence_24`: People frequently create charts via the chart editor in-game, and their Discord server is also decently active.
- `1063730_82978 review=4 sentence=sentence_2`: On the new world forums website I made a post about my reasons for quitting, and included in that post was fair criticism.
- `1063730_82978 review=4 sentence=sentence_4`: A day later my post was remove and my account was suspended because I "no longer am an active player" (referring to me mentioning im quitting).
- `1063730_82978 review=4 sentence=sentence_5`: This is a rather obvious attempt at censoring what was a popular thread containing discussion about the frankly atrocious state of the game.
- `1063730_82978 review=4 sentence=sentence_6`: I went to reddit to post about my ban and obvious attempt at censorship, that post was taken down within seconds.
- `1063730_82978 review=4 sentence=sentence_7`: I made a 2nd forums account and included in a new post screen shots of both the reddit post being taken down and my initial forum account being suspended.
- `1063730_82978 review=4 sentence=sentence_8`: My 2nd account forum was then suspended a 10 minutes after making the new post which was a PSA for the policing and censorship going on.
- `1063730_82978 review=4 sentence=sentence_9`: In that post I asked people to screen shot the post so that in case i was censored again, there was proof to share and raise the alarms.

### Topic 49 (195 docs)

Top words: rating, overall, rate, solid, overall game, nota, rate game, recommend, id, nota para

- `1017900_1112 review=5 sentence=sentence_8`: I rate this game: 7/10
- `1043260_3322 review=3 sentence=sentence_3`: В рейтинге этого режима по набранным очкам плаваю на позициях с 25 по 35.
- `1043810_6382 review=3 sentence=sentence_27`: Overall it's a solid 9/10.
- `1045720_3045 review=4 sentence=sentence_30`: Дякую за гру на 10/10 однозначно.
- `1055540_1264 review=3 sentence=sentence_13`: Overall I would give A Short Hike 4/5 stars
- `1060230_1945 review=3 sentence=sentence_22`: How then do I rate it?
- `107410_23399 review=10 sentence=sentence_8`: 10/10 will play in the next pandemic.
- `1084160_11858 review=3 sentence=sentence_23`: TLDR: 40yo man says JA3 good... and also that the internet is horrible.
- `1104450_8444 review=5 sentence=sentence_10`: Other than that, 10/10 can see this game getting big! <3
- `1105510_5640 review=4 sentence=sentence_9`: To be honest Haruka is still annoying so 9/10

### Topic 50 (193 docs)

Top words: remaster, remake, remastered, original, remasters, el, que, original game, mafia, version

- `1017900_1112 review=3 sentence=sentence_4`: the remake is really really nice and i have enjoyed it thoroughly.
- `1017900_1112 review=6 sentence=sentence_3`: While performance is obviously system specific, it makes no sense for a remastered modern version of a classic.
- `1030830_26263 review=3 sentence=sentence_2`: El mafia 2 original es un grandisimo juego que no falla en nada pero el remaster es una basura.
- `1030830_26263 review=5 sentence=sentence_1`: There's a lot this remaster does well, but as of launch, there is a whole lot that needs fixing.
- `1030830_26263 review=5 sentence=sentence_31`: If Hangar 13 fails to fix the three most important issues that they messed up from the original: the physics, flickering smoke, and eye animations, I will not buy the Mafia 1 Remastered or the future Mafia IV if they develop it.
- `1030830_26263 review=5 sentence=sentence_33`: But breaking things that already worked in the original Mafia II, putting shoddy bandaids on them or ignoring them entirely instead of fixing them, and releasing the remaster knowing these things are broken, is a pretty bad look that shows just how little they care.
- `1030830_26263 review=5 sentence=sentence_35`: And I know they shipped it out to another company to remaster; but since they obviously chose someone that wasn't up to the job, received a remaster that was visibly not up to par, and released it without fixing any of the new issues anyway, they might as well have broken it themselves.
- `1030830_26263 review=5 sentence=sentence_36`: After many of Mafia III's still-present issues and now this, Hangar 13 needs to prove to the community that they care about quality by fixing this remaster.
- `1030830_26263 review=6 sentence=sentence_3`: The Remaster of mafia 2 is, odd, to say the least, some textures got updated, most notably the roads and the character textures, as well as some more minor elements and Clemente's slaughterhouse seems to have gotten a bit of a repaint too.
- `1030840_80583 review=7 sentence=sentence_1`: (MDE) is a full remake of the original that came out back in 2002.

### Topic 51 (193 docs)

Top words: issues, game, mechanics, unfortunately, biggest, problem, issues game, problems game, gameplay, lot

- `1016790_1893 review=3 sentence=sentence_2`: Visually this game is extremely appealing, however, what lurks beneath the darkness is messy gameplay that doesn’t feel satisfying or addicting to play compared to other rogue-likes.•
- `1016950_4196 review=5 sentence=sentence_19`: There is no meat to this game, apart from the new stadiums (only one is available right now of course) everything is insanely low effort and rehashed material to the point that you start to wonder what they actually DID work on.
- `1028310_3994 review=4 sentence=sentence_9`: However, it is not a game without problems.
- `1055540_1264 review=3 sentence=sentence_12`: It's never game-breaking, but just janky sometimes.
- `1062810_3287 review=5 sentence=sentence_18`: It’s got a serious case of Marvel Midnight Suns Syndrome where a superb core gameplay system gets its attention diluted by a story and subsystems that are boring at best and bad at worst.
- `1073910_1574 review=3 sentence=sentence_2`: This game looks great and feels like it should be fun, but it has a couple issues that really hold it back.
- `1084160_11858 review=3 sentence=sentence_9`: There are a lot of systems in this game that seem either unfair or missing entirely (ambushing being probably the most common one I've seen people complaining about) but the fact is... most of them are in the game but the game does a very poor job of communicating how to use them.
- `1107790_3157 review=4 sentence=sentence_17`: One final note that wasn't important enough to mention earlier: the game is pretty amateurish in its presentation of medical procedure.
- `1120320_1755 review=4 sentence=sentence_3`: This game falls short in many areas.
- `1129580_41471 review=3 sentence=sentence_10`: First of all, let me talk about some problems with the game mechanism.

### Topic 52 (192 docs)

Top words: vr, vr games, vr game, best vr, vrchat, headset, vr vr, cvr, best, games

- `1012790_10355 review=8 sentence=sentence_1`: this game is THE BEST immersive survival game i have ever played, not just for VR, but as a WHOLE.
- `1049410_23830 review=7 sentence=sentence_4`: I can't recall the last time I was this captivated and transfixed with a video game; I'd go so far as to say this is the best non VR PC game I've played in 5 years.
- `1058830_3923 review=3 sentence=sentence_23`: 紧接着，就是适配VR设备与手柄控制器，通过VR投射出类似于节奏光剑的场景，更加有代入感。
- `1066890_9246 review=3 sentence=sentence_28`: Selling point #4: VR-support.
- `1066890_9246 review=3 sentence=sentence_29`: If you own a VR device, buying is basically a no-brainer, because there is no comparable performing sim for VR on the market.
- `1082680_1072 review=3 sentence=sentence_50`: Klar, die grundlegenden Techniken funktionieren und es muss nicht jedes Spiel völligen Realismus bieten, aber in VR macht Immersion halt viel aus, und die wird aufgrund vieler Abstriche in der Spielmechanik meiner Ansicht nach zerstört.
- `1104380_3590 review=3 sentence=sentence_3`: 从小时候的the room1开始追到the room3，虽然pc版一直迟过手机版，但是还是难挡补票的想法，the room4由网易代理叫迷逝，所以差点错过了，还好有人提醒我然后去查了一下，不然这作vr版的感受就大打折扣了
- `1104380_3590 review=3 sentence=sentence_6`: 这次的vr版，是我近期晚到的vr解密里面对眼睛最友好的游戏了，没有乱七八糟的抖动，舒服的操控和流畅的机关使用，恰到好处的提示。
- `1104380_3590 review=3 sentence=sentence_9`: 因为熟悉这个游戏，所以我以为这游戏对于我来说还是没啥难度的，但是我还是吃了不懂英语的亏，在第三关需要看英语石板的时候一脸懵逼，以及全程剧情没看懂，提示都看不动，完全不看攻略的情况下5小时通关vr版，初见不玩前4作大概6小时左右，4.操作：
- `1104380_3590 review=3 sentence=sentence_10`: 一开始我还觉得这游戏vr设计如果和pc版一样那肯定像个傻子一样，视野隔壁挂着一群东西，但是玩到后才发现，原来的目镜，背包，还有提示都用非常巧妙的融合在一体，拉提示的时候还感觉挺好玩的（虽然看不懂）5.直观感受：

### Topic 53 (188 docs)

Top words: cards, card, deck, decks, deckbuilding, draw, cards deck, trading cards, card games, hand

- `1022310_1594 review=3 sentence=sentence_3`: Deck building adds to the personal touch for the warbands.
- `1022310_1594 review=3 sentence=sentence_7`: The real life warbands (Models sold by Games Workshop) should include a code for unlocking them digitally much like Magic the Gatherings Planeswalker decks interaction with MTG:Arena.
- `1022310_1594 review=4 sentence=sentence_27`: I paid for Eyes of the Nine DLC, put together a deck offline using their standard cards but many of the cards I found were missing in-game.
- `1022310_1594 review=4 sentence=sentence_28`: I don't know why they would release a warband without making all cards available for use.
- `1022310_1594 review=4 sentence=sentence_40`: Also, sometimes you click on a card and, before putting it into play, the game treats it as your decision being made so you have to play it even though it technically hasn't been played.
- `1022310_1594 review=4 sentence=sentence_41`: This is where we often get stuck, because you are just looking at the ploy card or whatever, but you are forced to play it when you change your mind and want to put it back in your deck to play something else.
- `1022310_1594 review=4 sentence=sentence_42`: Then there is the card mechanic - it's not smooth like other CCGs such as Hearthstone.
- `1022310_1594 review=4 sentence=sentence_43`: You play a card, select a player for the card, then you have to play the card again!
- `1022310_1594 review=4 sentence=sentence_45`: They even ask you to select the player when the card is only for one player (ie your only wizard).
- `1037020_2760 review=3 sentence=sentence_14`: 当玩家用鼠标指向一张牌时，这张牌将会立即迅速放大，将卡牌完整而清晰地显示在玩家眼前。

### Topic 54 (188 docs)

Top words: crafting, craft, inventory, materials, item, items, resources, storage, gear, craft item

- `1028310_3994 review=4 sentence=sentence_13`: If you've ran out of books you can feasibly approach, for whatever reason that may be, and you also don't necessarily have the crafting materials required to do something else, it can be a boring slog of waiting for a new book from Oriflamme's, hoping that tomorrow's Weather is more beneficial, or that someone comes along to give you some Spintria for helping them with an objective.
- `1028310_3994 review=4 sentence=sentence_19`: In BOOK OF HOURS, you will be doing a LOT of crafting.
- `1028310_3994 review=4 sentence=sentence_26`: It would make Crafting SO MUCH less of an obtuse headache than it already is (since Hush House is RIDDLED with crafting stations, AND you are given an extreme gross of Skills), and it would make reading new books less of a tiresome process as you try and remember which books in your massive collection give you the proper Memories.3.)
- `1028310_3994 review=4 sentence=sentence_44`: Denzil is useless, the Rector is slightly less useless because he has desirable tags AND can use Candles, Sweet Bones hirelings are literally only ever useful because they don't exhaust Soul to acquire, money is worthless (I have so much saved up and there isn't even a furniture shoppe or general store!
- `1043260_3322 review=7 sentence=sentence_4`: çeşit çeşit gladyatörleri satın alıp kendi takımınızı oluşturuyor ve arenadan düşürdüğünüz itemlerle ya da bizzat kendinizin marketten satın alabileceği itemlerle bu gladyatörleri güçlendirebiliyorsunuz.
- `1043260_3322 review=7 sentence=sentence_12`: Bu şekilde gelişiyor; gold, odun, taş, demir kasarak hem gladyatörlerinizi hem de loncanızdaki gladyatör açmanızı sağlayan yapıları satın alabiliyor ve o yapıları geliştirip daha güçlü gladyatörler açmaya olanak sağlıyorsunuz.
- `1060230_1945 review=4 sentence=sentence_23`: Thankfully in Sapiens, any object that can be fashioned in anyway has a little lightbulb option on it and you must go out and find it (or craft it).
- `1062110_1737 review=3 sentence=sentence_25`: Another neat thing about Unhinged is its crafting system, in that as long as you know the recipe for something, you can make it at any crafting bench whenever you want, so if you want to spice up a run and do a gimmick, or just want to sequence break it over your knee for your current run, you can!
- `1069160_2309 review=5 sentence=sentence_19`: And there is limited crafting, in the form of cooking food and brewing potions.
- `1069160_2309 review=5 sentence=sentence_25`: By the end, you will probably be able to carry like 1k pounds collectively (you'll never become encumbered once you reach that threshold), but leading up to that, you will at some point have to manage your inventory a bit.

### Topic 55 (187 docs)

Top words: missions, mission, complete, misiones, missions missions, rewards, misses, campaign, missions feel, main missions

- `1009560_6684 review=4 sentence=sentence_12`: More and more missions couldn't be completed.
- `1016800_13368 review=3 sentence=sentence_9`: Missions are set in smaller sized mini zones, all with pretty distinct settings ranging from cramped streets with tall buildings to dense and foggy forests.
- `1016800_13368 review=4 sentence=sentence_15`: At the beginning of each day within the game you can assign your party members (and yourself) to go out on missions.
- `1030840_80583 review=7 sentence=sentence_8`: Given the linearity of the campaign, there's not much exploring to do at all during missions, even if some environments are a bit more open than others.
- `1043810_6382 review=3 sentence=sentence_22`: Even with every difficulty option set to maximum, you will most likely have very few issues until the last few missions in the game.
- `1051690_3450 review=3 sentence=sentence_15`: Technically, there are 25x as many levels, if you count all their variations (each level has a set of variations you can encounter.
- `1063420_3769 review=5 sentence=sentence_4`: This is a game that runs in self-contained missions.
- `1063420_3769 review=5 sentence=sentence_5`: These missions last from around 15 minutes to 2 hours, and nothing about your ship, character status, etc. will persist between them.
- `1072040_3577 review=5 sentence=sentence_10`: Die Einführung zu den Szenarien wird von einem russisch anmutenden Verbindungsoffizier gehalten, der recht emotionslos die Ziele der Mission erklärt.
- `1082680_1072 review=3 sentence=sentence_14`: Der "zweite Teil" sind die Storymissionen.

### Topic 56 (183 docs)

Top words: favorite, best, favorite games, played, favorite game, favourite, games, best games, ive played, best game

- `1016800_13368 review=3 sentence=sentence_28`: Chernobylite is by far one of my favorite games of 2021 and I cannot wait to see what else these devs pull off in the future.
- `1041920_1898 review=3 sentence=sentence_15`: Um dos meus jogos favoritos de 2020.
- `1055540_1264 review=7 sentence=sentence_2`: Scratched the itch that Night In The Woods left me with, which is my favorite game to date.
- `105600_54613 review=3 sentence=sentence_3`: Com toda a certeza é o melhor jogo que já joguei esse ano, valeu cada centavo, se você estiver procurando um jogo bom para passar o tempo esse é o jogo certo!
- `105600_54613 review=6 sentence=sentence_1`: Out of all the games I own, Terraria has to be my favorite.
- `105600_54613 review=9 sentence=sentence_2`: Let me say this is one of, if not the greatest games I have ever played.
- `1057090_26450 review=4 sentence=sentence_22`: 我只是在这里感慨自己最爱的游戏它的剧情完美落幕。
- `1060230_1945 review=3 sentence=sentence_4`: This is perhaps the greatest game I have played since Empire Earth 2.
- `1062110_1737 review=4 sentence=sentence_2`: One of the best games I have played.
- `1069530_1521 review=3 sentence=sentence_4`: This is my new favorite game of this year!

### Topic 57 (172 docs)

Top words: voice, voice acting, acting, actors, voice actors, voices, voiced, acted, characters, voice acted

- `1000030_2098 review=3 sentence=sentence_20`: The voice acting isn't going to win an award at the TGAs but it's cute and I still love the hums the customers make.
- `1036890_4908 review=3 sentence=sentence_29`: The voice acting by Wang's new (yeah dumb) actor is worse than the previous, and half the jokes feel forced and safe.
- `1040420_2914 review=3 sentence=sentence_22`: I know that there is one there, but unfortunately, I'm not much of a reader and this is one thing Hades definitely outdid with everything being voice-acted.
- `1042490_4762 review=3 sentence=sentence_30`: The island setting is beautiful and voice acting is superb.
- `1084160_11858 review=5 sentence=sentence_9`: Some aspects are exceptionally well done - the voice acting, the humor, the character design, the attention to details
- `1090630_8516 review=5 sentence=sentence_14`: Little touches like stage transitions, alternate outros and match-up based voice lines.
- `1101790_2018 review=3 sentence=sentence_10`: Voice acting is very good (it also features the Witcher 3 guy btw)#5.
- `1113560_24246 review=7 sentence=sentence_8`: EVERYONE is voiced too, which is pretty great!
- `1113560_24246 review=7 sentence=sentence_9`: Automata's characters were good too, but I personally thought Replicant's are significantly more memorable.JP vs EN dubbing:Both are done very well!
- `1113560_24246 review=7 sentence=sentence_10`: Weiss's and Kainé's voices stole the show for me personally.

### Topic 58 (168 docs)

Top words: issues, problems, problem, issue, needs, things, algumas, lot work, maybe lot, foram

- `1009560_6684 review=5 sentence=sentence_7`: They consistently message you, there is ALWAYS a problem.
- `1012790_10355 review=3 sentence=sentence_4`: There are some severe flaws with this approach however.
- `1016950_4196 review=3 sentence=sentence_4`: Here’s the *main* problems:
- `1020790_3591 review=5 sentence=sentence_1`: Antes de começar minha análise, não posso deixar de dizer que sim, já estive em todas as localidades da franquia STORM.
- `1022310_1594 review=4 sentence=sentence_7`: It's easier for me to list the problems currently...
- `1030210_11120 review=3 sentence=sentence_12`: Maybe, with a lot of work, it will become a worthy successor one day.
- `1035120_4085 review=7 sentence=sentence_9`: También en ciertos puntos podremos mejorar a nuestro personaje.
- `1049590_61216 review=4 sentence=sentence_17`: 앞으로도 이런 저런 논란 없이 흥하셨으면 합니다.
- `1063660_11863 review=4 sentence=sentence_10`: Despite this, there are some slight nitpicks I have.
- `1069640_4384 review=5 sentence=sentence_28`: Hinweis: Hier kommen einige Veränderungen je nach o.g. Kombination.

### Topic 59 (164 docs)

Top words: skills, skill, skill points, points, level, character, tree, levels, gear, passive

- `1016800_13368 review=4 sentence=sentence_8`: As you wander the radioactive wastes you accrue XP which accumulate into levels which award you with Skill Points.
- `1037020_2760 review=4 sentence=sentence_5`: 这可能包括了玩家本身技术的提升、角色属性的增幅，亦或是解锁的新武器、新道具。
- `1040420_2914 review=3 sentence=sentence_43`: Weapons don't upgrade or scale on their own.
- `1041720_6700 review=7 sentence=sentence_11`: Причем как класс (тут он называется "судьбой", в ремастере "предназначением"), так и выученные скиллы всегда можно сбросить - класс в любой момент, скиллы у непися за золото.•
- `1056640_23530 review=3 sentence=sentence_6`: Don't go all out on your armor and equipment, you will reach endgame needs pretty fast
- `1056640_23530 review=3 sentence=sentence_25`: ALSO unlike Hero, Phantom and Etoile CAN be used AS A SUBCLASS (but cannot HAVE a subclass).Getting classes to Lv.75 will unlock a title which will give you a PERMANENT Stat buff to your character across all classes, which at one point may be necessary for you to utilize.
- `1060230_1945 review=4 sentence=sentence_26`: Skills are earned through doing tasks and thankfully we are spared the usual background nonsense, where Bob has joined your village and has 25 points in leatherworking but only 3 in cooking, so he won't be in the kitchen any time soon.
- `1062110_1737 review=3 sentence=sentence_28`: Gears instead of just being a different word for consumables, instead have assorted effects that either have a timer that ticks down when activated or a certain amount of charges before it falls apart.
- `1062810_3287 review=5 sentence=sentence_23`: Whichever class you pick comes standard with three starting moves and a starting passive bonus, which you can accent with two additional moves and seven additional passives via items unlocked mid run.
- `1069160_2309 review=5 sentence=sentence_16`: From an RPG perspective, this has a nice amount of level up options via skill trees, with each skill having up to three tiers and as many as six upgrade slots.

### Topic 60 (159 docs)

Top words: bosses, boss, boss fights, fights, final boss, fight, boss fight, final, challenging, bosses game

- `1000410_2404 review=3 sentence=sentence_26`: -All three bosses in the game are some of the worst ones I've ever seen in an FPS.-The music is largely forgettable due to being generic ambience, not even being close to what Doom 64 had.
- `1020790_3591 review=3 sentence=sentence_12`: I can't speak on the history mode yet but I imagine that's just a collection of fights rather than there being any actual fun to be had in ways of big bosses.
- `1020790_3591 review=5 sentence=sentence_14`: História principal totalmente resumida, vários Boss Fighters foram removidos.
- `1040420_2914 review=3 sentence=sentence_37`: Most of the bosses and enemies seem very fair with their attacks and windows.
- `1056960_2523 review=3 sentence=sentence_7`: Even with boss fights, especially the last boss, it never left my side, it followed me, we revived eachother when needed, and it never went aimlessly run around, losing shared life or doing anything stupid.
- `1065310_9049 review=6 sentence=sentence_8`: Nearly every combat encounter has some kind of boss and depending on how you prefer to play Evil West, some bosses are harder than others.
- `1089090_19106 review=7 sentence=sentence_31`: Como joguei no modo mais difícil, as fases eram realmente desafiadoras, principalmente os BOSSES, que por sinal são muito bem feitos.
- `1102190_20493 review=4 sentence=sentence_13`: At first, it seems like a no-brainer to just shove all your big dudes into one lane and kill things before they can even get further in, but the bosses challenge you not to do that.
- `1102190_20493 review=4 sentence=sentence_19`: You fight the same bosses in the same order every run, but they have different crests that change the mechanics of the fights significantly.
- `1102190_20493 review=4 sentence=sentence_20`: However, it still feels like the same boss, and you might have to change your tactics turn-to-turn, but you don't have to build significantly differently to handle them.

### Topic 61 (158 docs)

Top words: money, dinheiro, waste, waste money, sobrando, se tiver, dinheiro sobrando, compre se, debt, tiver dinheiro

- `1001270_4904 review=5 sentence=sentence_23`: No matter how much money you spend at once (be it 100 or 20000) the animation of the total number of cash you have goes down at the same pace.
- `1003590_7381 review=5 sentence=sentence_5`: Why spend money trying to improve when it's already perfect?
- `1009560_6684 review=5 sentence=sentence_9`: I can’t pay all at once and have to keep coming back to pay the bills.
- `1009560_6684 review=5 sentence=sentence_11`: Just make a designated day for all tenants to pay their bills.
- `1009560_6684 review=5 sentence=sentence_15`: How the hell do I owe MORE as I’m trying to pay it off?
- `1009560_6684 review=5 sentence=sentence_17`: I was so annoyed when I took out a 100k loan and had 60k and couldn’t just give 50k so I would owe less and have less interest.
- `1009560_6684 review=5 sentence=sentence_21`: Just throw money in the rooms.
- `1015500_833 review=5 sentence=sentence_9`: Spend your money elsewhere.
- `1040200_11371 review=3 sentence=sentence_15`: 크레딧에 제 이름이 있을 수도?ㅈ..
- `1062520_16450 review=3 sentence=sentence_18`: Your cash flow will be just fine.

### Topic 62 (149 docs)

Top words: save, saves, save game, save file, autosave, file, saved, progress, manual, manual save

- `1001270_4904 review=5 sentence=sentence_11`: You cannot save midday.
- `1066890_9246 review=3 sentence=sentence_7`: You will notice that this game allows the user to save slides more than others, and this is essentially correct.
- `1084640_2165 review=3 sentence=sentence_25`: Ещё в игре нет ручных сохранений, в отличие от других визуальных новелл, тут если уж накосячил, то накосячил, не плачь.
- `1104330_2060 review=4 sentence=sentence_3`: [기본 팁]• 자동저장을 지원하지만 시간간격이 20분이다.• 맵은 대/중/소 크기가 있다.
- `1104330_2060 review=4 sentence=sentence_16`: > 윈도우 해상도를 FHD로 변경 후 게임을 껏다 저장 후 4k에서 재시작함.• 캐릭터가 계속 다른 도구를 기본으로 장착함.
- `1118310_7122 review=10 sentence=sentence_2`: Если есть важные настройки, то рекомендую сделать бэкап
- `1119700_2053 review=4 sentence=sentence_13`: In the regular game you must save when you exit, it saves automatically every turn, and there's only 1 save file.
- `1119700_2053 review=4 sentence=sentence_15`: This makes it play much like Faster Than Light or Slay the Spire; in 2 hours you either made it to the end of the round or died, so savescumming would be silly, yet the other most popular complaint people seem to have is the lack of multiple save files and ability to reload a couple turns back.
- `1126320_7150 review=3 sentence=sentence_42`: Auto-Save, auto-scroll between scenes, quick save, etc.
- `1137460_10661 review=10 sentence=sentence_7`: Oyunda en nefret ettiğim kısım ise save balonunu patlatıp ilgili save hakkını aldığınızda bunun sizi sadece 1 kez save yerine götürüyor olması.

### Topic 63 (146 docs)

Top words: plant, crops, grow, crop, farming, plants, ready, harvest, seeds, lets

- `1040070_2071 review=3 sentence=sentence_10`: (eg, you can buy a wheat field and then build a windmill and bakery and end up with wheat, flour, then bread.)
- `1060230_1945 review=3 sentence=sentence_17`: You see, feeding your villagers is insanely important to prevent starvation and unfortunately, this task is manual.
- `1060230_1945 review=3 sentence=sentence_18`: Which means you must manually command villagers to gather apples, peaches, and other fruits or manually order them to hunt specific animals each and every time warm weather comes around or they will die out in the cold weather.
- `1062520_16450 review=3 sentence=sentence_31`: You could get a daily to water crops but haven't seen the farming person yet, or somebody could ask you for brewed food and you haven't unlocked the keg yet.- (As of writing)
- `1069640_4384 review=5 sentence=sentence_17`: Außerdem übernehmen sie die Farming- und Crafting-Aufgaben.
- `1069640_4384 review=5 sentence=sentence_36`: Augen auf was in dem Gebiet so wächst und lebt, alles was in guter Menge Verfügbar ist hat Vorrang, anderes versuchen sie auch zu besorgen, kann aber dauern, kann auch nicht klappen.
- `1087760_2041 review=3 sentence=sentence_27`: For instance, seeds can be thrown in special “energy pools” to make plants grow, then acting as platforms, or explosive plants thrown onto debris to clear a path.
- `1104330_2060 review=4 sentence=sentence_31`: [연구 팁]• 목공예와 농업은 필수 연구다.
- `1115690_10619 review=4 sentence=sentence_12`: There is rarely ever a good reason to turn down peasants, when they ask you for things.
- `1115690_10619 review=4 sentence=sentence_23`: Of course, if you've said yes to every single peasant like I said you should, you can afford them all without any problems but, just in case you can't, then you can take comfort in knowing that you don't need a single one of them, ever.

### Topic 64 (142 docs)

Top words: speed limit, limit, speed, locomotive, railway, lcd, car, ui, signal, arrow

- `1030840_80583 review=6 sentence=sentence_1`: 这个游戏告诉我们：第一，要懂得弯道超车，你要减速要学会忍耐，不要一股脑跟傻卵一样速度拉满，有时候，减速得到的是安全又高效的超车，急性子只会撞在干草垛上然后听后面超过你的啥b嘲讽一句:嘿，菜逼。
- `1030840_80583 review=6 sentence=sentence_2`: 第二，你如果开始是最后一名，不代表你永远是最后一名，或许下一秒，第一名的车就报废了，同样的，你开到了第一个，不代表你永远是第一名，没到终点谁也说不清，没准一会你就翻车了然后被后面车全超了。
- `1097150_92230 review=7 sentence=sentence_13`: 腿放在终点线前没有动，他也承认，我先到达终点线，他不知道腿放在终点线前面，他承认我先到达终点线啊！
- `1119730_16960 review=5 sentence=sentence_19`: 7，希望增加一些木质小推车的种类，现有车辆的速度太快，并没有档位限制，希望增加档位的按钮。
- `1178490_18052 review=3 sentence=sentence_4`: 最后一关那辆车不是人开的，我只能说碰碰车都比那个好开。。。
- `1222680_4350 review=4 sentence=sentence_4`: ④本作漂移失速严重，漂移竞速党慎入⑤本作没有丰田，所以没有AE86，头文字D迷慎入⑥心想还原所有速度与激情里的车的玩家也许会失望，因为车并不全，而且针对某些车的外观改装也没有，尤其是韩的rx7是别想了⑦本作百余辆车中有小一半是老爷车，豪车党慎入⑧本作警车过于凶猛，以至于大部分情况都不能靠速度和车技逃脱，而只能通过跳板或是bug点逃脱，可玩性大打折扣，属于减分项，玻璃心玩家慎入
- `1222680_4350 review=4 sentence=sentence_19`: 每晚经验结算是惹火等级乘以你的罚款，理论上多跑几个图、多惹惹警察就能快速升级，但请注意晚上维修站只能修两次车，新手不要贪得无厌，否则被抓今晚就顶级白忙活
- `1222680_4350 review=4 sentence=sentence_22`: (2)漂移赛当然需要漂移专用的轮胎、悬挂、差速器，但你不一定非要全换成漂移用的，混搭有奇效;
- `1222680_4350 review=4 sentence=sentence_23`: (3)轮胎这东西，并不是加性能分最多的就最好，如果你能忍受高性能分带来的负面影响
- `1248130_7880 review=4 sentence=sentence_8`: 7全新马牌和其他轮胎品牌加入--------------------------------------------------------------------------------------------------------------------------

### Topic 65 (135 docs)

Top words: good decent, bad, decent bad, beautiful good, good, decent, beautiful, bad beautiful, bad bad, schlecht

- `1029550_11568 review=3 sentence=sentence_7`: Very good☑ Good☐ Not too bad☐ Bad☐
- `1082430_15462 review=6 sentence=sentence_6`: Beautiful☐ Good☐ Decent☐ Bad☐
- `1082430_15462 review=6 sentence=sentence_32`: Average☐ Good☐ Lovely☑
- `1126320_7150 review=3 sentence=sentence_46`: The scale’s order goes like this: Massive ♥♥♥, Huge ♥♥♥, ♥♥♥, Neutral, Chick, Huge Chick & Massive Chick.
- `1170880_1083 review=7 sentence=sentence_1`: Simplesmente Incrível, Bonito, Fofo, Maravilhoso por mais que seja curto é muito bem feito.
- `1201240_11231 review=6 sentence=sentence_2`: Beautiful☐ Good☑ Decent☐ Bad☐
- `1201540_3395 review=6 sentence=sentence_2`: Красиво☐ Хорошо☑ Приемлемо☐ Плохо☐
- `1218250_4574 review=3 sentence=sentence_2`: What Is This?▢ Acceptable▢ Good🟩 Great▢ Beautiful▢ Masterpiece~ SOUND/MUSIC ~ (Using Headphones)▢ Bad▢ Nothing Special▢ Good🟩 Great▢ Beautiful▢
- `1253920_14141 review=7 sentence=sentence_1`: ✖️ Horrível.✖️ Ruim.✖️ Aceitavel.✖️ Lindo.✔️ Fantástico.✖️ Nenhum.✖️ Ruim.✖️ Aceitavel.✖️ Boa.✔️ Maravilhosa.✖️ Horrível. ✖️ Ruim.✖️ Aceitável. ✖️ Linda.
- `1253920_14141 review=7 sentence=sentence_2`: ✔️ Fantástica.✖️ Bugada.✖️ Repetitiva/Entediante.✖️ Frustrante.

### Topic 66 (135 docs)

Top words: needs, hope, future, suggestions, potential, game, forward, things need, game potential, game better

- `1040200_11371 review=3 sentence=sentence_17`: 최대한 빨리 수정하고 여러분들에게 피해 안 끼치겠습니다다들 즐거운 게임 플레이가 되셨으면 좋겠습니다!+참고로 저는 회사 직원이 아니라 일반 유저일 뿐입니다
- `1055540_1264 review=6 sentence=sentence_12`: I will be paying attention to the creators of this game for their future endeavors.
- `1066890_9246 review=3 sentence=sentence_5`: After countless, countless of physics changes, the game still has not quite arrived where it wants to be; however it is very close.
- `1066890_9246 review=3 sentence=sentence_21`: Sometimes, it still feels like the workload is huge, and the game definitely has some way to go until it is a finished product - but there is a lot of fun to be had with what's on offer already, and there's fun to be had in seeing it evolve.
- `1105500_5697 review=3 sentence=sentence_27`: Сложность игры и корректировка HP стала более сбалансированной, хоть и не идеальна как будет в последующих играх.
- `1118200_831 review=3 sentence=sentence_7`: This game could've been way better with a little developer effort.
- `1119730_16960 review=4 sentence=sentence_1`: jogo é maravilhoso, só falta colocar algumas atualizações como:
- `1133760_1983 review=4 sentence=sentence_7`: Já não basta o fiasco do Chrono Trigger , eles conseguiram fazer algo PIOR.
- `1137300_3360 review=3 sentence=sentence_48`: If Frogware can iron out the problems mentioned here, I believe the next game will be awesome because you can see what they were going for by making it an open world, but it just wasn’t fleshed out in the finished product.
- `1148760_3360 review=7 sentence=sentence_1`: This game is good, but it really needs a primer, so I will give you one now:

### Topic 67 (134 docs)

Top words: abandoned, updates, development, devs, update, developers, developer, game, players, dev

- `1020790_3591 review=5 sentence=sentence_31`: Bom, os desenvolvedores não revisaram o jogo (CTRL+C/CTRL+V) no Storm 4, não foi brincadeira.
- `1030210_11120 review=3 sentence=sentence_5`: The game feels like an abandoned beta.
- `1030830_26263 review=5 sentence=sentence_38`: The community's good will matters more than they might think; Mafia III's sales were underwhelming because of all its issues and that can happen again.
- `1058650_7389 review=7 sentence=sentence_5`: Players would be better, but devs can't make players.
- `1063730_82978 review=4 sentence=sentence_10`: This game is in an atrocious state, and the developers dont care to communicate.
- `1063730_82978 review=5 sentence=sentence_15`: En fin, el juego está muriendo o para otros ya murió gracias a la mala ejecución de los devs y la otra parte de la culpa se la llevan los jugadores tóxicos que no dudaron nunca en aprovechar los errores para sentirse bien una vez en sus miserables vidas.
- `1097150_92230 review=9 sentence=sentence_7`: I really feel betrayed with the decisions of the game devs because I always supported the game even in awful situations like the cheating problem, now to see everything *fall* apart, very sad to know that money can change anyone and make the worst mistakes ever.
- `1106840_11729 review=3 sentence=sentence_2`: Meanwhile WB Games and TT Games massive dev team dropped features at the last minute that they had previously promised, released a heavily bugged game and haven't patched or communicated with their players for weeks.
- `1128000_5722 review=7 sentence=sentence_2`: Released the game, had some updates right at the beginning & then left for a few years. - You don't screw over your player base two times and think that's okay.
- `1128000_5722 review=8 sentence=sentence_6`: Steam release version is worse and apparently he just kept his money and abandoned the game.

### Topic 68 (133 docs)

Top words: dishes, cooking, cleaning, customers, food, clean, toilet, restaurant, ingredients, flip

- `1000030_2098 review=3 sentence=sentence_25`: The new Auto Serve key allows the breakneck speed of CSD2 without the heartbreak of accidentally pushing the key for something you started cooking.
- `1000030_2098 review=3 sentence=sentence_29`: Making some of the more complicated dishes was hard enough without having to hunt down dishes ready to go.
- `1000030_2098 review=3 sentence=sentence_30`: One still has to navigate through pages of ingredients as in CSD2, but the removal of side dish juggling and addition of Auto Serve makes this is less stressful.
- `1000030_2098 review=3 sentence=sentence_40`: It's nice not having foods 'taken away' from you once you reach a certain point like in CSD, your access to food will remain.
- `1000030_2098 review=3 sentence=sentence_46`: The food truck contrastingly has you constantly cooking.
- `1001270_4904 review=4 sentence=sentence_15`: İstasyon olayı kalkarsa eğer mesela bir kişi çorba tatlı kebap 3lü sipariş veremiyor sadece menüden tek bir yemek seçiyor bu değişirse oyuna renk katar.
- `1001270_4904 review=4 sentence=sentence_21`: Yapmazsanız şişiniz gidiyor.Şişten toplama animasyonu gelmesi lazım.Restoran tuvaletini müşteriler kullanabilmeli.
- `1001270_4904 review=5 sentence=sentence_2`: fantastic. - You get to manually cook the meals step-by-step after you buy the ingredients at the stores nearby (NOT online). - You get to decorate your restaurant the way you want, moving EVERYTHING in the store besides the doors, windows, and walls. - You get to interact with the customers yourself or you can hire waiters to do it for you. - You get to choose where to store your ingredients. - You get to prepare food and put it in the fridge...
- `1036240_1798 review=4 sentence=sentence_1`: asks, and answers, one question: What if the secret herbs and spices at a similar, but legally distinct, fast food restaurant was actually ?
- `1036240_1798 review=5 sentence=sentence_2`: I think it just needs a few things:1: Duplicate worker to make that much quicker when hiring someone.

### Topic 69 (132 docs)

Top words: humor, jokes, funny, laugh, humour, joke, make laugh, hilarious, silly, references

- `1034140_19644 review=6 sentence=sentence_10`: The humour, and the writing in general are great-
- `1034140_19644 review=6 sentence=sentence_11`: Waifu's are hilarious and incredibly hot-
- `1040070_2071 review=4 sentence=sentence_20`: Everything's goofy, and everything goofy is avoidable too.
- `1040200_11371 review=5 sentence=sentence_22`: Just random items in a safe or a drawer that make you laugh if you get the reference.
- `1049410_23830 review=6 sentence=sentence_3`: My response to that was, "Yeah, I guess it's not as funny as three of the GREATEST comedies of the last thirty years.
- `1084160_11858 review=4 sentence=sentence_9`: humor and story-telling thrown in.
- `1086940_51540 review=3 sentence=sentence_23`: 你很有幽默感，可惜并不是所有人都认同这一点。
- `1089980_30650 review=6 sentence=sentence_5`: The fails and jokes can give you a quick smirk or laugh but that's pretty much it.
- `1089980_30650 review=6 sentence=sentence_6`: Still, pretty fun if you want a quick laugh.
- `1113000_73340 review=4 sentence=sentence_30`: There are scenes that will make you cry from laughter, and others that will make your heart race as you see the events unfold on your screen.

### Topic 70 (123 docs)

Top words: bugs, bugs game, game breaking, bug, gamebreaking, buggy, breaking, issues, breaking bugs, mnh

- `1000410_2404 review=3 sentence=sentence_13`: This issue is only remedied in the final levels of the game.
- `1022310_1594 review=4 sentence=sentence_16`: Lots and lots of bugs in the game still.
- `1030830_26263 review=5 sentence=sentence_16`: This is a very noticeable issue that wasn't in the original game, it's distracting and needs fixing ASAP.3.
- `1063420_3769 review=7 sentence=sentence_7`: It's still got a lot of bugs, regularly enter scenarios where one player can't interact with specific things- Lootboxes!
- `1114150_7984 review=3 sentence=sentence_52`: After some more time playing, the issues I've found have still been pretty pervasive.
- `1150440_11610 review=3 sentence=sentence_4`: The game is, however, still hindered by its many, numerous bugs.
- `1150440_11610 review=3 sentence=sentence_20`: I also love Battlefleet Gothic 1/2 from their studio, but like this game, both came out with numerous game-breaking bugs.
- `1161580_19510 review=7 sentence=sentence_11`: Is the game still buggy?
- `1173220_2309 review=3 sentence=sentence_1`: This game seriously needs to be patched.
- `1202130_10155 review=5 sentence=sentence_9`: As for game play bugs, I found a couple times that the units abilities wouldn't fire or there were some pathfinding issues, especially with the Marauder Suits.

### Topic 71 (121 docs)

Top words: achievements, achievement, score, leaderboardsranks isnt, high score, care leaderboardsranks, leaderboardsranks, isnt necessary, necessary progress, achievements wait

- `1000030_2098 review=3 sentence=sentence_32`: Delicious ratings are less of a hassle to collect as they simply replace your Great ratings once you hit a certain combo of perfect dishes - no more worrying about having side dishes prepped in your incredibly important Holding Stations.
- `1022310_1594 review=4 sentence=sentence_19`: It is an area that would keep me coming back more but given a lot of games end up being conceded I basically play for hours without even getting any XP to even unlock the basic cosmetics.
- `1029550_11568 review=3 sentence=sentence_19`: Only if u care about leaderboards/ranks☐ Isn't necessary to progress☐
- `1029550_11568 review=6 sentence=sentence_15`: - Records the lead run of a driver instead of running against a player live in PvP.
- `1061910_14713 review=4 sentence=sentence_21`: De plus, pour les énervés du scoring, des leaderboards avec un classement mondial pourront vous motiver à tenter la "run" parfaite.
- `1061910_14713 review=5 sentence=sentence_15`: The way they do the Score Multiplier.
- `1061910_14713 review=5 sentence=sentence_21`: Since it tracks your score and has a leader board the multiplier makes sense.
- `1061910_14713 review=5 sentence=sentence_26`: HOWEVER, EVERYTHING is tied to the multiplier.
- `1061910_14713 review=5 sentence=sentence_30`: Is tied to the multiplier.
- `1121640_5548 review=3 sentence=sentence_40`: Although a lot of content has been added since the Demo version, the an achievement system is still absent currently, making the replayability a bit low

### Topic 72 (121 docs)

Top words: animations, animation, animaes, animated, smooth, theyre, animations bit, animations look, sprites, stiff

- `1016950_4196 review=5 sentence=sentence_18`: I’ve seen many animations that are identical to the previous edition.
- `1020790_3591 review=5 sentence=sentence_15`: Não tem cutscene, não tiveram a mínima coragem de pelo menos fazer uma arte mais bonita, simplesmente colocaram uma imagem do anime, no clássico é pior ainda, tiraram a imagem do Naruto de 2002.
- `1022310_1594 review=3 sentence=sentence_8`: I would also like to see more animations as some attacks/blocks go unnoticed.
- `1034140_19644 review=3 sentence=sentence_8`: 단순 애니메이션, 3인칭 3D 회전 기능 사용 불가 (SFM 수준)- 얼엑 기준 3 캐릭터의 잡다한 신들을 포함하면 20개 정도, 그것도 3-4개를 제외하면 대부분 중요하지 않은 씬- 해금은 각 캐릭터의 레벨업, 혹은 호감도 선물을 통해서 이루어집니다-
- `1045720_3045 review=4 sentence=sentence_24`: Тепер портрети персонажів під час діалогів анімовані згідно із сучасними тенденціями, моделі/спрайти виглядають значно промальованішими, оточення надзвичайно органічне, деталізоване, багатошарове.
- `1058650_7389 review=3 sentence=sentence_11`: Animation: still clunky, but I think they're going to improve.
- `1072040_3577 review=5 sentence=sentence_21`: Die Animation der Einheiten ist recht schwach.
- `1083880_2779 review=4 sentence=sentence_36`: 姫を含めた全てのアニメーションを見られる可能性があるがいかんせんランダムで解放されるので大金が必要な茨の道だ
- `1090630_8516 review=5 sentence=sentence_10`: Stunning animations.
- `1097130_2125 review=4 sentence=sentence_3`: A animações super fluidas e bem feitas.

### Topic 73 (121 docs)

Top words: human fall, oa, fall flat, oo, flat, fall, human

- `1030210_11120 review=5 sentence=sentence_6`: Даже Вольсен себя чувствует в разы лучше, а он в двое старше чем Торч 3.Думаю нет смысла питать ложные надежды.
- `1030210_11120 review=5 sentence=sentence_38`: Чаще всего там будет просто переход в следующую кишку.
- `1043260_3322 review=3 sentence=sentence_8`: Это я в ней залип в надежде отыскать что же там нового добавили.
- `1072040_3577 review=4 sentence=sentence_12`: А я приду и посмотрю".
- `107410_23399 review=6 sentence=sentence_1`: Тупое гoвнo тупого гoвнa.
- `107410_23399 review=9 sentence=sentence_1`: Тупое гавно, тупого гавна, саааааааааааааааааааамая худшая трата за всю жизнь, надеюсь разрабы поперхнулись каждым рублем который уплачен за это "творение" которое делали макаки с руками которые ломали им с самого рождения
- `1170950_12464 review=3 sentence=sentence_37`: Потому что каждый понимает что за свои слова придется отвечать!-
- `1194930_3778 review=5 sentence=sentence_8`: Кстати, гладиаторов можно пытать, но не стоит этого делать слишком часто.
- `1227690_5948 review=6 sentence=sentence_20`: Мне, как любителю старого Макса Пейна, вообще нереально зашло.
- `1232580_2000 review=3 sentence=sentence_8`: В общем, как говорится, "будем посмотреть".

### Topic 74 (120 docs)

Top words: art, art style, style, arte, artistic, visuals, artstyle, beautiful, painting, work art

- `1069160_2309 review=4 sentence=sentence_8`: both are paintings, both are beautiful, but the expressionist conveys such emotion, such intangible substance, that the photorealistic painter can not.
- `1114220_3476 review=4 sentence=sentence_18`: Art style: I like it, you may not.
- `1121640_5548 review=3 sentence=sentence_15`: 特別的美術風格、有趣的背景故事、漂亮的偽2D視角同時兼顧美感與效能
- `1121640_5548 review=3 sentence=sentence_42`: Unique art style, interesting background story, beautiful pseudo-2D, fulfill both aesthetics and performance
- `1121640_5548 review=3 sentence=sentence_43`: Not much to say, just look at the steam page and you'll appreciate the beautiful artworks!
- `1121640_5548 review=3 sentence=sentence_44`: The art style hits my heart as soon as I saw it
- `1123450_3125 review=3 sentence=sentence_12`: Heck, it even brought me out of weeks-long art block with a fun drawing system - enough limitations to stop me from nitpicking every element of my work, enough freedom to make my creations feel mine.
- `1123450_3125 review=3 sentence=sentence_18`: In my eyes, that's what sets a work of art apart.
- `1123450_3125 review=4 sentence=sentence_10`: The expectation of human reaction can get in the way of passion and conveying emotion far more than the barrier between an artist and the canvas they choose to paint on.
- `1123450_3125 review=4 sentence=sentence_12`: When I was growing up, I was unabashedly proud of anything and everything I created, drew, painted or transformed.

### Topic 75 (119 docs)

Top words: check run, run paint, paint, requirements check, check, pc requirements, requirements, teens, kids teens, teens adults

- `1029550_11568 review=3 sentence=sentence_10`: Check if you can run paint☐
- `1049590_61216 review=5 sentence=sentence_1`: ===Graficos===🔲Atari🔲140p🔲Aceitável✅Bom🔲Ótimo🔲The Witcher 3🔲Red Dead Redemption II
- `1049590_61216 review=5 sentence=sentence_2`: 🔲Cada frame foi pintado por Da Vinci===Requisitos===🔲Qualquer coisa que conduza energia🔲Uma calculadora com 1 pilha✅Um celular🔲Um i5 com uma GTX 750TI🔲No minímo uma GTX 1660🔲Pc com CPU e GPU muito boas
- `1082430_15462 review=6 sentence=sentence_8`: Paint.exe---{Gameplay}---☑
- `1082430_15462 review=6 sentence=sentence_18`: Lizards---{PC Requirements}---☑
- `1106840_11729 review=7 sentence=sentence_9`: I'm now deaf---{ Audience }---☐ Kids☑ Teens☑ Adults☑ Grandma---{ PC Requirements }---☐ Check if you can run paint☐ Potato☑ Decent☐ Fast☐
- `1145960_5728 review=4 sentence=sentence_2`: Güzel☐ İyi☐Normal☐ Kötü☐ Uzun Süre Bakma☑ Paint.exe---{Oynanış}---☐ Çok İyi☑ İyi☐ Normal☐ Eh İşte☐ Duvarlara Bakmak Daha Eğlenceli☐
- `1190970_8810 review=4 sentence=sentence_30`: You can paint pixel art on the walls if you want to.-The Flipper tool allows easy style changes, copying, and duplication.
- `1201240_11231 review=6 sentence=sentence_9`: I'm now deaf---{ Audience }---☐ Kids☑ Teens☐ Adults☐ Grandma---{ PC Requirements }---☑ Check if you can run paint☐ Potato☐ Decent☐ Fast☐
- `1201270_5269 review=4 sentence=sentence_9`: Kebutuhan PC 🔵✅ Bisa buka aplikasi Paint?✅ Kentang🔲 Rata-Rata🔲 Kencang🔲

### Topic 76 (119 docs)

Top words: wa, open eyes, ga, eyes, logic open, wo suru, suru, wo, pole, pal

- `1003590_7381 review=5 sentence=sentence_33`: Everything, for the King.
- `1015500_833 review=8 sentence=sentence_9`: NO ROAD TO WRESTLEMANIA!!
- `1058650_7389 review=8 sentence=sentence_15`: L'ordi de ma grand mère☐
- `1119730_16960 review=4 sentence=sentence_20`: ENTÃO VAI BOMBAR MAIS AINDA
- `1145960_5728 review=4 sentence=sentence_10`: Beyin İster☐ Öğrenmesi
- `1150640_4109 review=4 sentence=sentence_4`: Mais lento que uma tartaruga paralítica
- `1150640_4109 review=4 sentence=sentence_23`: Novas Animações De Invocações
- `1179080_6470 review=8 sentence=sentence_2`: + This ain't no place for you, preacher + SUFFER
- `1179080_6470 review=8 sentence=sentence_8`: + NOBODY'S COMING, PRIEST
- `1180380_1256 review=3 sentence=sentence_1`: Боги одного проекта, или БАНЫ навсегда

### Topic 77 (118 docs)

Top words: ui, interface, menus, ui ui, menu, ui elements, sampai, user interface, dalam, user

- `1000030_2098 review=3 sentence=sentence_8`: The menus and level select were obtuse to the point where I had to click blindly to navigate.
- `1016950_4196 review=5 sentence=sentence_41`: The confusing UI is one thing, but it wasn’t until the menu navigation that I started to lose my temper.
- `1022310_1594 review=4 sentence=sentence_38`: It feels like the UX person isn't a player of the game because they've not laid out the information and UI well enough.
- `1022310_1594 review=4 sentence=sentence_47`: 99% of the errors are actually our own doing, but they are made because of the poor UX.
- `1045720_3045 review=4 sentence=sentence_26`: Інтерфейс уже вичищений, місця для тексту повно, кнопки розташовані зручно, усе логічно й просто.
- `1062520_16450 review=4 sentence=sentence_11`: Very good, fast menu/UI, good pacing
- `1118200_831 review=3 sentence=sentence_2`: - Controls and menus are not user friendly.
- `1135300_5180 review=4 sentence=sentence_7`: The worst UI I've seen in a long time.
- `1135300_5180 review=5 sentence=sentence_17`: Отвратительный интерфейс.
- `1138660_5670 review=6 sentence=sentence_11`: Needs improvement on UI/some card tweaks.

### Topic 78 (118 docs)

Top words: xcom, xcoms, xcom2, reboot xcoms, reboot, like xcom, phoenix point, strategy, phoenix, tactics

- `1043810_6382 review=3 sentence=sentence_9`: There is no XCOM "5% chance to miss" in this game.
- `1043810_6382 review=3 sentence=sentence_25`: Anyone who's cut their teeth on other games like Into The Breach or XCOM will breeze through a lot of this game.
- `1084160_11858 review=3 sentence=sentence_21`: This game is not a "X-Com asset flip", literally five seconds with the game will tell you that (literally one feature, the "enemy reposition", and the cover shields is all it has in common)
- `1127700_4259 review=3 sentence=sentence_26`: 多くのマップで敵の視界と移動経路をチェックしつつ隠密行動が求められるというゲーム性であり、その点はXCOMを想起させる。
- `1184050_7067 review=3 sentence=sentence_1`: Gears Tactics is not XCOM.
- `1184050_7067 review=3 sentence=sentence_2`: This is a good thing, I think - though the game is in a genre that is inspired and overshadowed by XCOM, it demonstrates that all tactics games don't need to be mere reskins to be good.
- `1184050_7067 review=3 sentence=sentence_3`: That's not to say the general ideas won't be immediately accessible to veterans of XCOM and similar games.
- `1184050_7067 review=3 sentence=sentence_7`: What makes Gears Tactics truly interesting are its deviations from the XCOM formula: every subtly adjusted mechanic feels like a genuine step forward for the genre as a whole, a thoughtful refinement on the original template.
- `1184050_7067 review=3 sentence=sentence_25`: XCOM and those games closely modeled after it often place the player in situations where they have to carefully creep up on foes and lay out ambushes and traps in order to draw out overwhelmingly powerful foes.
- `1184050_7067 review=3 sentence=sentence_27`: Where XCOM evokes horror and a perpetual sense of being outclassed, Gears puts you firmly in the boots of a team of badasses gunning down Locusts by the dozens.

### Topic 79 (112 docs)

Top words: nasa, nasa spare, ask nasa, spare computer, spare, computer, ask, computer difficulty, just press, press

- `1029550_11568 review=3 sentence=sentence_14`: Ask NASA if they have a spare computer---{ 🕹️Difficulty🕹️ }---☐ Just press 'W'☑ Easy☐
- `1049890_11254 review=3 sentence=sentence_28`: 你可能需要一台超超超超级计算机---{难度}---☐
- `1069530_1521 review=5 sentence=sentence_18`: Narita Boy disassemble the underlaying graphical computations of the related Trichroma button in order to bypass protolcols to allow mainframe access".
- `1082430_15462 review=6 sentence=sentence_21`: Ask NASA if they have a spare computer---{Difficulity}---☑
- `1106840_11729 review=7 sentence=sentence_11`: Ask NASA if they have a spare computer---{ Difficulty }---☐ Just press 'W'☑
- `1110910_8180 review=3 sentence=sentence_15`: Necesitas▢ Computadora de la NASA~ DURACION DE JUEGO ~▢ ya termino ? (0-2 Hours)🟥 Cortina (2-20 Hours) (Solo la historia base)🟥 Ni muy largo (20-50 Hours) (Con todos los Logros)▢ Larguito (50-200 Hours)▢ Infinito casi (200-... Hours)▢ Multiplayer/Sin Fin~ ES DIVERTIDO ? ~▢ Divertido como Chupar un Clavo▢ Dificil de divertrise con esto▢
- `1135810_2645 review=4 sentence=sentence_16`: The electric battery powering Elon Musk's Telsa.
- `1145960_5728 review=4 sentence=sentence_6`: Kertenkeleler---{Bilgisayar Gereksinimleri}---☑
- `1145960_5728 review=4 sentence=sentence_8`: NASA’dan Bilgisayar İste---{Zorluk}---☐ Sadece “a” Tuşuna Bas☐
- `1158940_9072 review=7 sentence=sentence_17`: BİLGİSAYAR-GEREKSİNİMLERİ ~🔲

### Topic 80 (112 docs)

Top words: dark souls, dark, souls, grind, souls grind, difficult dark, grind grind, difficult, grind difficult, grind dark

- `1029550_11568 review=3 sentence=sentence_18`: Dark Souls---{ 📈Grind📉 }---☑ Nothing to grind☐
- `1082430_15462 review=6 sentence=sentence_26`: Difficult☐ Dark Souls---{Grind}---☑ Nothing to grind☐
- `1106840_11729 review=7 sentence=sentence_15`: Difficult☐ Dark Souls---{ Grind }---☐ Nothing to grind☐
- `1108590_3491 review=3 sentence=sentence_43`: A história de Eldest Souls é clichê e simplória: um mundo devastado pela Desolação, uma maldição dos Deuses Antigos para punir os humanos.
- `1123050_5364 review=5 sentence=sentence_3`: I’m an avid fan of Souls and Soulslikes who loves a more gritty, obscured story.
- `1123050_5364 review=6 sentence=sentence_4`: Unlike many other Souls clones, GRIME manages to recreate the underlying horror and unease of Dark Souls without copying its rotting medieval aesthetic.
- `1145960_5728 review=4 sentence=sentence_14`: Dark Souls---{Hikaye}---☐
- `1201240_11231 review=6 sentence=sentence_15`: Difficult☐ Dark Souls---{ Grind }---☐ Nothing to grind☐
- `1201270_5269 review=4 sentence=sentence_5`: Susah banget kek b#ngs#t cem' dark souls🔵 Grind 🔵✅ Tidak Perlu Grind🔲 Grind cuman buat flexxing (leaderboard/ranks)🔲 Sedikit Perlu Grind🔲 Rata Rata harus Grind🔲 Harus Grind🔲 Grind Sampe Mampus🔵 Waktu Game 🔵🔲 Singkat (<6h)✅ Pendek (<15h)🔲 Rata Rata (15h-30h)🔲 Panjang (30h-100h)🔲 Panjang Sekali (>100h)🔲 Tak terbatas (∞)🔵 Grafik 🔵🔲 Lebih buruk dari game retro 8/16-Bit🔲 Dibawah standar🔲 Rata Rata✅ Grafik gak penting🔲 Bagus🔲 Sangat Bagus
- `1226470_5509 review=4 sentence=sentence_11`: predatory this company is, Shadow arena was first a content for BLACK DESERT ONLINE.

### Topic 81 (106 docs)

Top words: pvp, pve, pvp pvp, pve pvp, pvp f2p, server, gzw, consegu, pvpve, pvp game

- `1063730_82978 review=5 sentence=sentence_3`: Ni hablar de los exploits de PvP donde muchos perdieron territorios que ganaron legalmente y que nunca recibieron una respuesta de Amazon para banearlos.
- `1063730_82978 review=5 sentence=sentence_12`: Por mi parte jugué absolutamente todo ya que estuve más de la mitad de mis horas en un server brasilero y no conseguí hablar con ningún hispanohablante así que me las arregle solo, hice todas las misiones principales/secundarias, me cole en runs de portales y elites, subí oficios, vendí, tradie, conseguí mi moneda y mi equipamiento, disfruté todo hasta que me pasé a un server ES y empezó el ""PvP"" donde entré a varias guilds, conseguí un grup...
- `1063730_82978 review=6 sentence=sentence_4`: I got my character to max level, started making friends in-game, and got hooked on the PvP.
- `1128860_5460 review=4 sentence=sentence_34`: On top of it, nothing prevents you from going and playing realism mode instead in "PVP classic".
- `1150640_4109 review=4 sentence=sentence_7`: RanqueadoPVP tem uma contagem regressiva muito curta (pra quem quer combar de verdade), parece que você está jogando batata quente, você fica extremamente apreensivo com medo do tempo acabar;
- `1227280_3088 review=4 sentence=sentence_14`: 던전이나 보스가 크게 변하지 않기에 pvp에 흥미가 없으시다면 클리어 후 리플레이 동기가 떨어지는 편입니다.
- `1254120_29820 review=3 sentence=sentence_1`: Forced PVP after lvl 30....
- `1254120_29820 review=3 sentence=sentence_2`: you can toggle off pvp but this only makes it so YOU cant attack others but others can still attack you......
- `1262240_11390 review=7 sentence=sentence_1`: 作为PVZ系列的一个衍生作品，无论是在美工设计还是在游戏剧情上都别具一格，特色鲜明。
- `1262240_11390 review=7 sentence=sentence_11`: *pve的剧情设计，玩法，解密元素等都十分优秀，剧情风格可以说是匠心独运。

### Topic 82 (96 docs)

Top words: fnaf, birthday, stayed, witch, fish, happy, came, nice, love, gameplay

- `1089090_19106 review=3 sentence=sentence_1`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⠀⠀⠀⢠⣾⣧⣤⡖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠋⠀⠉⠀⢄⣸⣿⣿⣿⣿⣿⣥⡤⢶⣿⣦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡆⠀⠀⠀⣙⣛⣿⣿⣿⣿⡏⠀⠀⣀⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⠷⣦⣤⣤⣬⣽⣿⣿⣿⣿⣿⣿⣿⣟⠛⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠋⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣠⣶⣶⣶⣿⣦⡀⠘⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠈⢹⡏⠁⠀⠀⠀⠀⠀⢀⣿⡏⠉⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡆⠀⢀⣿⡇⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣟⡘⣿⣿⣃⠀⠀⠀⣴⣷⣀⣸⣿⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⠹⣿⣯⣤⣾⠏⠉⠉⠉⠙⠢⠀⠈⠙⢿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣄⠛⠉⢩⣷⣴⡆⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣀⡠⠋⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
- `1089090_19106 review=4 sentence=sentence_2`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⠀⠀⠀⢠⣾⣧⣤⡖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠋⠀⠉⠀⢄⣸⣿⣿⣿⣿⣿⣥⡤⢶⣿⣦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡆⠀⠀⠀⣙⣛⣿⣿⣿⣿⡏⠀⠀⣀⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⠷⣦⣤⣤⣬⣽⣿⣿⣿⣿⣿⣿⣿⣟⠛⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠋⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣠⣶⣶⣶⣿⣦⡀⠘⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠈⢹⡏⠁⠀⠀⠀⠀⠀⢀⣿⡏⠉⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡆⠀⢀⣿⡇⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣟⡘⣿⣿⣃⠀⠀⠀⣴⣷⣀⣸⣿⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⠹⣿⣯⣤⣾⠏⠉⠉⠉⠙⠢⠀⠈⠙⢿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣄⠛⠉⢩⣷⣴⡆⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣀⡠⠋⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
- `1089090_19106 review=6 sentence=sentence_1`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⠀⠀⠀⢠⣾⣧⣤⡖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠋⠀⠉⠀⢄⣸⣿⣿⣿⣿⣿⣥⡤⢶⣿⣦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡆⠀⠀⠀⣙⣛⣿⣿⣿⣿⡏⠀⠀⣀⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⠷⣦⣤⣤⣬⣽⣿⣿⣿⣿⣿⣿⣿⣟⠛⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠋⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣠⣶⣶⣶⣿⣦⡀⠘⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠈⢹⡏⠁⠀⠀⠀⠀⠀⢀⣿⡏⠉⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡆⠀⢀⣿⡇⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣟⡘⣿⣿⣃⠀⠀⠀⣴⣷⣀⣸⣿⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⠹⣿⣯⣤⣾⠏⠉⠉⠉⠙⠢⠀⠈⠙⢿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣄⠛⠉⢩⣷⣴⡆⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣀⡠⠋⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
- `1089090_19106 review=8 sentence=sentence_1`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⠀⠀⠀⢠⣾⣧⣤⡖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠋⠀⠉⠀⢄⣸⣿⣿⣿⣿⣿⣥⡤⢶⣿⣦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡆⠀⠀⠀⣙⣛⣿⣿⣿⣿⡏⠀⠀⣀⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⠷⣦⣤⣤⣬⣽⣿⣿⣿⣿⣿⣿⣿⣟⠛⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠋⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣠⣶⣶⣶⣿⣦⡀⠘⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠈⢹⡏⠁⠀⠀⠀⠀⠀⢀⣿⡏⠉⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡆⠀⢀⣿⡇⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣟⡘⣿⣿⣃⠀⠀⠀⣴⣷⣀⣸⣿⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⠹⣿⣯⣤⣾⠏⠉⠉⠉⠙⠢⠀⠈⠙⢿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣄⠛⠉⢩⣷⣴⡆⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣀⡠⠋⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀
- `1089090_19106 review=9 sentence=sentence_1`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⣿⠀⠀⠀⢠⣾⣧⣤⡖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⠋⠀⠉⠀⢄⣸⣿⣿⣿⣿⣿⣥⡤⢶⣿⣦⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⡆⠀⠀⠀⣙⣛⣿⣿⣿⣿⡏⠀⠀⣀⣿⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⠀⠙⠻⠷⣦⣤⣤⣬⣽⣿⣿⣿⣿⣿⣿⣿⣟⠛⠿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣴⠋⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⡆⠀⠀⠀⠀⠀⠀⣠⣶⣶⣶⣿⣦⡀⠘⣿⣿⣿⣿⣿⣿⣿⣿⠿⠋⠈⢹⡏⠁⠀⠀⠀⠀⠀⢀⣿⡏⠉⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡆⠀⢀⣿⡇⠀⠀⠀⠀⠀⠀⢸⣿⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣟⡘⣿⣿⣃⠀⠀⠀⣴⣷⣀⣸⣿⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⠹⣿⣯⣤⣾⠏⠉⠉⠉⠙⠢⠀⠈⠙⢿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣄⠛⠉⢩⣷⣴⡆⠀⠀⠀⠀⠀⠀⠀⠀⠋⠀⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣀⡠⠋⠈⢿⣇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⠿⠛⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀
- `1113560_24246 review=6 sentence=sentence_1`: ⠀⢱⡈⡆⠀⠀⠀⠀⠀⠀⠀⠀⢣⠸⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⢇⢿⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⡆⢷⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⣾⣧⡀⠀⣀⠤⠤⢤⣀⡀⠀⠀⠀⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣷⠊⠀⠀⠀⠀⠀⠙⢦⠀⠀⠀⠀⢠⣴⣿⣿⡟⠁⠈⠙⢍⠉⠀⢠⢦⠀⢠⡄⠀⠰⣇⠀⠀⠀⢰⣿⣿⣿⠏⠀⠀⠀⠀⢨⠀⠐⣿⠀⢧⡘⣷⣄⠰⣿⡀⠀⠀⠀⢺⣿⣿⣧⠀⠀⠀⠀⠀⣩⠀⢸⢳⡀⠀⠛⠈⢿⣦⠙⢧⡄⠀⠀⠀⠈⢿⣿⣿⣿⣦⢤⢈⣶⢋⣿⡞⠶⠁⠀⠀⢰⣿⣟⣷⡄⡙⠦⡀⠀⠀⠀⠀⠀⠉⠻⡿⠋⠀⣽⣥⣿⡿⣧⠔⢢⠀⠀⠘⡟⣿⣾⡋⢢⣄⠈⡆⠀⠀⠀⠀⣀⠤⠀⢈⢲⡾⠋⠀⠉⢻⢾⡈⠀⠀⠀⣰⣁⣽⡄⠹⡀⡛⢷⣅⠀⠀⠀⠀⠀⠀⠠⣲⣻⣾⠲⣄⠀⡎⣹⣿⣿⣿⣿⣿⣯⡀⠈⢣⡙⢴⢄⠿⣝⡦⠄⠀⠀⠀⠀⠀⡐⡩⣹⡿⠁⠀⡿⣷⣯⣿⣿⣿⣿⣿⣿⣿⣿⣧⢈⠱⣄⣷⣕⢮⠛⢶⣄⣂⠀⠀⠀⢀⣞⡜⣹⡟⠀⠀⢸⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⡶⠞⢿⣿⣧⠠⣀⠛⢝⠛⢿⣶⣄⠀⠀⢀⡾⠋⢠⡟⠀⠀⠀⡏⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⡻⣤⣀⣈⣿⠻⣧⡈⠻⢶⣴⡄⠈⢿⡻⢦⠀⠀⡼⠁⠀⣾⠀⠀⠀⠀⡇⣿⣿⣿⢹⣿⣿⣿⣿⣿⣿⣿⣇⣷⢻⣿⣿⣿⣀⠉⠻⣄⠀⡉⢻⣄⠀⢻⠀⠁⠀⡰⠁⠀⢠⡇⠀⠀⠀⠀⣇⢿⣿⣿⡌⣿⣿⣿⣿⣿⣿⣿⣗⣿⣮⣿⣿⣿⡟⣆...
- `1129540_7403 review=4 sentence=sentence_2`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⢀⣀⣀⣀⣀⡀⠤⠄⠒⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣀⠄⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠋⠉⠈⠉⠉⠉⠉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⢏⣴⣿⣷⠀⠀⠀⠀⠀⢾⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣟⣾⣿⡟⠁⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣷⢢⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣟⠀⡴⠄⠀⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⠟⠻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⢴⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⣿⣿⣁⡀⠀⠀⢰⢠⣦⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⡄⠀⣴⣶⣿⡄⣿⣿⡋⠀⠀⠀⠎⢸⣿⡆⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠗⢘⣿⣟⠛⠿⣼⣿⣿⠋⢀⡌⢰⣿⡿⢿⡀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⡇⠀⢸⣿⣿⣧⢀⣼⣿⣿⣷⢻⠄⠘⠛⠋⠛⠃⠀⠀⠀⠀⠀⢿⣧⠈⠉⠙⠛⠋⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣧⠀⠈⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⠀⠀⠀⠀⢀⢃⠀⠀⢸⣿⣿⣿⣿⣿⣿⡿⠀⠴⢗⣠⣤⣴⡶⠶⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀...
- `1129540_7403 review=6 sentence=sentence_2`: ⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀ ⡇⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀ ⡇⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁ ⡇⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀ ⡇⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀ ⡇⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀⠀ ⡇⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⠀ ⡇⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀ ⡇⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀ ⡇⢸⠀⠀⠀⠀⢠⠃⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀ ⡇⢸⠀⠀⠀⠀⢸⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀ ⢸⠀
- `1129540_7403 review=7 sentence=sentence_2`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⢀⣀⣀⣀⣀⡀⠤⠄⠒⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣀⠄⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠋⠉⠈⠉⠉⠉⠉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⢏⣴⣿⣷⠀⠀⠀⠀⠀⢾⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣟⣾⣿⡟⠁⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣷⢢⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣟⠀⡴⠄⠀⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⠟⠻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⢴⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⣿⣿⣁⡀⠀⠀⢰⢠⣦⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⡄⠀⣴⣶⣿⡄⣿⣿⡋⠀⠀⠀⠎⢸⣿⡆⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠗⢘⣿⣟⠛⠿⣼⣿⣿⠋⢀⡌⢰⣿⡿⢿⡀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⡇⠀⢸⣿⣿⣧⢀⣼⣿⣿⣷⢻⠄⠘⠛⠋⠛⠃⠀⠀⠀⠀⠀⢿⣧⠈⠉⠙⠛⠋⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣧⠀⠈⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⠀⠀⠀⠀⢀⢃⠀⠀⢸⣿⣿⣿⣿⣿⣿⡿⠀⠴⢗⣠⣤⣴⡶⠶⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀...
- `1129540_7403 review=11 sentence=sentence_3`: ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⢀⣀⣀⣀⣀⡀⠤⠄⠒⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣀⠄⠊⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠋⠉⠈⠉⠉⠉⠉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⢏⣴⣿⣷⠀⠀⠀⠀⠀⢾⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣟⣾⣿⡟⠁⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣷⢢⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣟⠀⡴⠄⠀⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⠟⠻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⢴⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⣿⣿⣁⡀⠀⠀⢰⢠⣦⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⡄⠀⣴⣶⣿⡄⣿⣿⡋⠀⠀⠀⠎⢸⣿⡆⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠗⢘⣿⣟⠛⠿⣼⣿⣿⠋⢀⡌⢰⣿⡿⢿⡀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⡇⠀⢸⣿⣿⣧⢀⣼⣿⣿⣷⢻⠄⠘⠛⠋⠛⠃⠀⠀⠀⠀⠀⢿⣧⠈⠉⠙⠛⠋⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣧⠀⠈⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⠀⠀⠀⠀⢀⢃⠀⠀⢸⣿⣿⣿⣿⣿⣿⡿⠀⠴⢗⣠⣤⣴⡶⠶⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀...

### Topic 83 (94 docs)

Top words: fell, death, die, died, dead, lyorum, ento, burried, exato momento, exato

- `1000360_6765 review=3 sentence=sentence_22`: When I die in Hellish Quart, it's my fault, and that feels GREAT.
- `1012790_10355 review=6 sentence=sentence_5`: I excitedly hurried up the creaking, rusty body of the crane.
- `1012790_10355 review=6 sentence=sentence_9`: Lost my grip and fell a hundred feet, died on impact.
- `1040420_2914 review=3 sentence=sentence_39`: Either my overconfidence and haste, paired with some precarious trap and red floor placement, met me with a cruel fate.
- `1063660_11863 review=4 sentence=sentence_12`: The conditions for dying are also weirdly binary.
- `1069650_1910 review=3 sentence=sentence_35`: 엄폐 100퍼센트를 껴도 두세방이면 그대로 궤멸당합니다.
- `1112890_4627 review=4 sentence=sentence_19`: You cannot even kill yourself, and any attempts to do so to escape results in absolutely nothing.
- `1112890_4627 review=4 sentence=sentence_20`: You do not fall; you gently float down.
- `1208260_4734 review=3 sentence=sentence_38`: Seu vizinho chama a polícia.
- `1218250_4574 review=5 sentence=sentence_15`: I got trapped a couple of times and had to strafe my way out a couple of times.- maybe petty but I expected some kind of payoff in the end.

### Topic 84 (90 docs)

Top words: lumina, nazi, beloved, award, start new, porn, wife, buy game, thank, thought

- `1097150_92230 review=8 sentence=sentence_1`: ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠛⠉⠉⠄⠄⠄⠉⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠄⢀⣠⣶⣶⣶⣶⣤⡀⠄⠄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡏⠄⠄⣾⣿⢿⣿⣿⡿⢿⣿⡆⠄⠄⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠄⠄⢿⣇⣸⣿⣿⣇⣸⡿⠃⠄⠄⠸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠄⠄⠄⠄⠄⠉⠛⠛⠛⠛⠉⠄⠄⠄⠄⠄⠄⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠁⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⢿⣿⣿⣿⣿⣿⣿⡟⠄⠄⠄⠠⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠈⢿⣿⣿⣿⣿⡟⠄⠄⠄⢠⣆⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⣧⠄⠄⠄⠈⢿⣿⣿⣿⡇⠄⠄⠄⣾⣿⡀⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⠄⢰⣿⣧⠄⠄⠄⠘⣿⣿⣿⣇⠄⣰⣶⣿⣿⣿⣦⣀⡀⠄⠄⠄⠄⠄⠄⠄⢀⣠⣴⣿⣿⣿⣶⣆⠄⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠄⠄⢸⣿⠇⠄⠄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣤⣴⣾⣿⣶⣤⣤⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
- `1113560_24246 review=3 sentence=sentence_1`: ⠄⠄⠄⠄⢠⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣯⢻⣿⣿⣿⣿⣆⠄⠄⠄⠄⠄⣼⢀⣿⣿⣿⣿⣏⡏⠄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣧⢻⣿⣿⣿⣿⡆⠄⠄⠄⠄⡟⣼⣿⣿⣿⣿⣿⠄⠄⠄⠈⠻⣿⣿⣿⣿⣿⣿⣿⣇⢻⣿⣿⣿⣿⠄⠄⠄⢰⠃⣿⣿⠿⣿⣿⣿⠄⠄⠄⠄⠄⠄⠙⠿⣿⣿⣿⣿⣿⠄⢿⣿⣿⣿⡄⠄⠄⢸⢠⣿⣿⣧⡙⣿⣿⡆⠄⠄⠄⠄⠄⠄⠄⠈⠛⢿⣿⣿⡇⠸⣿⡿⣸⡇⠄⠄⠈⡆⣿⣿⣿⣿⣦⡙⠳⠄⠄⠄⠄⠄⠄⢀⣠⣤⣀⣈⠙⠃⠄⠿⢇⣿⡇⠄⠄⠄⡇⢿⣿⣿⣿⣿⡇⠄⠄⠄⠄⠄⣠⣶⣿⣿⣿⣿⣿⣿⣷⣆⡀⣼⣿⡇⠄⠄⠄⢹⡘⣿⣿⣿⢿⣷⡀⠄⢀⣴⣾⣟⠉⠉⠉⠉⣽⣿⣿⣿⣿⠇⢹⣿⠃⠄⠄⠄⠄⢷⡘⢿⣿⣎⢻⣷⠰⣿⣿⣿⣿⣦⣀⣀⣴⣿⣿⣿⠟⢫⡾⢸⡟⠄.⠄⠄⠄⠄⠻⣦⡙⠿⣧⠙⢷⠙⠻⠿⢿⡿⠿⠿⠛⠋⠉⠄⠂⠘⠁⠞⠄⠄⠄⠄⠄⠄⠄⠄⠈⠙⠑⣠⣤⣴⡖⠄⠿⣋⣉⣉⡁⠄⢾⣦⠄⠄⠄⠄⠄⠄⠄⠄
- `1113560_24246 review=5 sentence=sentence_1`: ⠄⠄⠄⠄⢠⣿⣿⣿⣿⣿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣯⢻⣿⣿⣿⣿⣆⠄⠄⠄⠄⠄⣼⢀⣿⣿⣿⣿⣏⡏⠄⠹⣿⣿⣿⣿⣿⣿⣿⣿⣧⢻⣿⣿⣿⣿⡆⠄⠄⠄⠄⡟⣼⣿⣿⣿⣿⣿⠄⠄⠄⠈⠻⣿⣿⣿⣿⣿⣿⣿⣇⢻⣿⣿⣿⣿⠄⠄⠄⢰⠃⣿⣿⠿⣿⣿⣿⠄⠄⠄⠄⠄⠄⠙⠿⣿⣿⣿⣿⣿⠄⢿⣿⣿⣿⡄⠄⠄⢸⢠⣿⣿⣧⡙⣿⣿⡆⠄⠄⠄⠄⠄⠄⠄⠈⠛⢿⣿⣿⡇⠸⣿⡿⣸⡇⠄⠄⠈⡆⣿⣿⣿⣿⣦⡙⠳⠄⠄⠄⠄⠄⠄⢀⣠⣤⣀⣈⠙⠃⠄⠿⢇⣿⡇⠄⠄⠄⡇⢿⣿⣿⣿⣿⡇⠄⠄⠄⠄⠄⣠⣶⣿⣿⣿⣿⣿⣿⣷⣆⡀⣼⣿⡇⠄⠄⠄⢹⡘⣿⣿⣿⢿⣷⡀⠄⢀⣴⣾⣟⠉⠉⠉⠉⣽⣿⣿⣿⣿⠇⢹⣿⠃⠄⠄⠄⠄⢷⡘⢿⣿⣎⢻⣷⠰⣿⣿⣿⣿⣦⣀⣀⣴⣿⣿⣿⠟⢫⡾⢸⡟⠄⠄⠄⠄⠄⠄⠻⣦⡙⠿⣧⠙⢷⠙⠻⠿⢿⡿⠿⠿⠛⠋⠉⠄⠂⠘⠁⠞⠄⠄⠄⠄⠄⠄⠄⠄⠈⠙⠑⣠⣤⣴⡖⠄⠿⣋⣉⣉⡁⠄⢾⣦⠄⠄⠄⠄⠄⠄⠄⠄
- `1137460_10661 review=9 sentence=sentence_2`: ⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠋⠉⠈⠉⠉⠉⠉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⢏⣴⣿⣷⠀⠀⠀⠀⠀⢾⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣟⣾⣿⡟⠁⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣷⢢⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣟⠀⡴⠄⠀⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⠟⠻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⢴⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⣿⣿⣁⡀⠀⠀⢰⢠⣦⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⡄⠀⣴⣶⣿⡄⣿⣿⡋⠀⠀⠀⠎⢸⣿⡆⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠗⢘⣿⣟⠛⠿⣼⣿⣿⠋⢀⡌⢰⣿⡿⢿⡀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⡇⠀⢸⣿⣿⣧⢀⣼⣿⣿⣷⢻⠄⠘⠛⠋⠛⠃⠀⠀⠀⠀⠀⢿⣧⠈⠉⠙⠛⠋⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣧⠀⠈⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⠀⠀⠀⠀⢀⢃⠀⠀⢸⣿⣿⣿⣿⣿⣿⡿⠀⠴⢗⣠⣤⣴⡶⠶⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡸⠀⣿⣿⣿⣿⣿⣿⣿⡀⢠⣾⣿⠏⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠉⠀⣿⣿⣿⣿⣿⣿⣿⣧⠈⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿...
- `1146630_6863 review=4 sentence=sentence_1`: ⣿⣿⣿⡇⢩⠘⣴⣿⣥⣤⢦⢁⠄⠉⡄⡇⠛⠛⠛⢛⣭⣾⣿⣿⡏⣿⣿⣿⡇⠹⢇⡹⣿⣿⣛⣓⣿⡿⠞⠑⣱⠄⢀⣴⣿⣿⣿⣿⡟⣿⣿⣿⣧⣸⡄⣿⣪⡻⣿⠿⠋⠄⠄⣀⣀⢡⣿⣿⣿⣿⡿⠋⠘⣿⣿⣿⣿⣷⣭⣓⡽⡆⡄⢀⣤⣾⣿⣿⣿⣿⣿⡿⠋⠄⢨⡻⡇⣿⢿⣿⣿⣭⡶⣿⣿⣿⣜⢿⡇⡿⠟⠉⠄⠸⣷⡅⣫⣾⣿⣿⣿⣷⣙⢿⣿⣿⣷⣦⣚⡀⠄⠄⢉⣾⡟⠙⠈⢻⣿⣷⣅⢻⣿⣿⣿⣿⣿⣶⣶⡆⠄⡀⠄⢠⣿⣿⣧⣀⣀⣀⣀⣼⣿⣿⣿⡎⢿⣿⣿⣿⣿⣿⣿⣇⠄⠄⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⢇⣎⢿⣿⣿⣿⣿⣿⣿⣿⣶⣶⠄⠄⠻⢿⣿⣿⣿⣿⣿⣿⣿⢟⣫⣾⣿⣷⡹⣿⣿⣿⣿⣿⣿⣿⡟⠄⠄⠄⠄⢮⣭⣍⡭⣭⡵⣾⣿⣿⣿⡎⣿⣿⣌⠻⠿⠿⠿⠟⠋⠄⠄⠄⠄⠈⠻⣿⣿⣿⣿⣹⣿⣿⣿⡇⣿⣿⡿⠄⠄⣀⣴⣾⣶⡞⣿⣿⣿⣿⣿⣿⣿⣾⣿⡿⠃⣠⣾⣿⣿⣿⣿⣿⣹⣿⣿⣿⣿⣿⡟⣹⣿⣳⡄
- `1273710_8900 review=6 sentence=sentence_1`: ⠄⠄⠄⣾⣿⠿⠿⠶⠿⢿⣿⣿⣿⣿⣦⣤⣄⢀⡅⢠⣾⣛⡉⠄⠄⠄⠸⢀⣿⠄⠄⢀⡋⣡⣴⣶⣶⡀⠄⠄⠙⢿⣿⣿⣿⣿⣿⣴⣿⣿⣿⢃⣤⣄⣀⣥⣿⣿⠄⠄⢸⣇⠻⣿⣿⣿⣧⣀⢀⣠⡌⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⣿⣿⣿⠄⢀⢸⣿⣷⣤⣤⣤⣬⣙⣛⢿⣿⣿⣿⣿⣿⣿⡿⣿⣿⡍⠄⠄⢀⣤⣄⠉⠋⠄⣼⣖⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⢇⣿⣿⡷⠶⠶⢿⣿⣿⠇⢀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣷⣶⣥⣴⣿⢀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢸⣿⣦⣌⣛⣻⣿⣿⣧⠙⠛⠛⡭⠅⠒⠦⠭⣭⡻⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠘⣿⣿⣿⣿⣿⣿⣿⣿⡆⠄⠄⠄⠄⠄⠄⠄⠄⠹⠈⢋⣽⣿⣿⣿⣿⣵⣾⠃⠄⠘⣿⣿⣿⣿⣿⣿⣿⣿⠄⣴⣿⣶⣄⠄⣴⣶⠄⢀⣾⣿⣿⣿⣿⣿⣿⠃⠄⠄⠄⠈⠻⣿⣿⣿⣿⣿⣿⡄⢻⣿⣿⣿⠄⣿⣿⡀⣾⣿⣿⣿⣿⣛⠛⠁⠄
- `1274290_6350 review=3 sentence=sentence_1`: ⠄⠄⠄⣾⣿⠿⠿⠶⠿⢿⣿⣿⣿⣿⣦⣤⣄⢀⡅⢠⣾⣛⡉⠄⠄⠄⠸⢀⣿⠄⠄⢀⡋⣡⣴⣶⣶⡀⠄⠄⠙⢿⣿⣿⣿⣿⣿⣴⣿⣿⣿⢃⣤⣄⣀⣥⣿⣿⠄⠄⢸⣇⠻⣿⣿⣿⣧⣀⢀⣠⡌⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⣿⣿⣿⠄⢀⢸⣿⣷⣤⣤⣤⣬⣙⣛⢿⣿⣿⣿⣿⣿⣿⡿⣿⣿⡍⠄⠄⢀⣤⣄⠉⠋⠄⣼⣖⣿⣿⣿⣿⣿⣿⣿⣿⣿⢿⣿⣿⣿⣿⣿⢇⣿⣿⡷⠶⠶⢿⣿⣿⠇⢀⠘⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣽⣿⣿⣿⡇⣿⣿⣿⣿⣿⣿⣷⣶⣥⣴⣿⢀⠈⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⢸⣿⣦⣌⣛⣻⣿⣿⣧⠙⠛⠛⡭⠅⠒⠦⠭⣭⡻⣿⣿⣿⣿⣿⣿⣿⣿⡿⠃⠘⣿⣿⣿⣿⣿⣿⣿⣿⡆⠄⠄⠄⠄⠄⠄⠄⠄⠹⠈⢋⣽⣿⣿⣿⣿⣵⣾⠃⠄⠘⣿⣿⣿⣿⣿⣿⣿⣿⠄⣴⣿⣶⣄⠄⣴⣶⠄⢀⣾⣿⣿⣿⣿⣿⣿⠃⠄⠄⠄⠈⠻⣿⣿⣿⣿⣿⣿⡄⢻⣿⣿⣿⠄⣿⣿⡀⣾⣿⣿⣿⣿⣛⠛⠁⠄⠄
- `1280770_16276 review=6 sentence=sentence_2`: 2⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠋⠉⠈⠉⠉⠉⠉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⢏⣴⣿⣷⠀⠀⠀⠀⠀⢾⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣟⣾⣿⡟⠁⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣷⢢⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣟⠀⡴⠄⠀⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⠟⠻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⢴⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⣿⣿⣁⡀⠀⠀⢰⢠⣦⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⡄⠀⣴⣶⣿⡄⣿⣿⡋⠀⠀⠀⠎⢸⣿⡆⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠗⢘⣿⣟⠛⠿⣼⣿⣿⠋⢀⡌⢰⣿⡿⢿⡀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⡇⠀⢸⣿⣿⣧⢀⣼⣿⣿⣷⢻⠄⠘⠛⠋⠛⠃⠀⠀⠀⠀⠀⢿⣧⠈⠉⠙⠛⠋⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣧⠀⠈⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⠀⠀⠀⠀⢀⢃⠀⠀⢸⣿⣿⣿⣿⣿⣿⡿⠀⠴⢗⣠⣤⣴⡶⠶⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡸⠀⣿⣿⣿⣿⣿⣿⣿⡀⢠⣾⣿⠏⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠉⠀⣿⣿⣿⣿⣿⣿⣿⣧⠈⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰...
- `1282150_3166 review=5 sentence=sentence_2`: ⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠛⠛⠛⠋⠉⠈⠉⠉⠉⠉⠛⠻⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⢿⣿⣿⣿⣿⣿⣿⣿⣿⡏⣀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣤⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⢏⣴⣿⣷⠀⠀⠀⠀⠀⢾⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠈⣿⣿⣿⣿⣟⣾⣿⡟⠁⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣷⢢⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣟⠀⡴⠄⠀⠀⠀⠀⠀⠀⠙⠻⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⠟⠻⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⢴⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⣿⣿⣁⡀⠀⠀⢰⢠⣦⠀⠀⠀⠀⠀⠀⠀⠀⢀⣼⣿⣿⣿⣿⣿⡄⠀⣴⣶⣿⡄⣿⣿⡋⠀⠀⠀⠎⢸⣿⡆⠀⠀⠀⠀⠀⠀⣴⣿⣿⣿⣿⣿⣿⣿⠗⢘⣿⣟⠛⠿⣼⣿⣿⠋⢀⡌⢰⣿⡿⢿⡀⠀⠀⠀⠀⠀⠙⠿⣿⣿⣿⣿⣿⡇⠀⢸⣿⣿⣧⢀⣼⣿⣿⣷⢻⠄⠘⠛⠋⠛⠃⠀⠀⠀⠀⠀⢿⣧⠈⠉⠙⠛⠋⠀⠀⠀⣿⣿⣿⣿⣿⣿⣿⣧⠀⠈⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠟⠀⠀⠀⠀⢀⢃⠀⠀⢸⣿⣿⣿⣿⣿⣿⡿⠀⠴⢗⣠⣤⣴⡶⠶⠖⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⡸⠀⣿⣿⣿⣿⣿⣿⣿⡀⢠⣾⣿⠏⠀⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠛⠉⠀⣿⣿⣿⣿⣿⣿⣿⣧⠈⢹⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣰⣿...
- `1290490_8124 review=11 sentence=sentence_1`: ⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⠿⠿⠿⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠟⣉⡥⠶⢶⣿⣿⣿⣿⣷⣆⠉⠛⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢡⡞⠁⠀⠀⠤⠈⠿⠿⠿⠿⣿⠀⢻⣦⡈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠘⡁⠀⢀⣀⣀⣀⣈⣁⣐⡒⠢⢤⡈⠛⢿⡄⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⢀⣼⣿⣿⣿⣿⣿⣿⣿⣿⣶⣄⠉⠐⠄⡈⢀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⢠⣿⣿⣿⣿⡿⢿⣿⣿⣿⠁⢈⣿⡄⠀⢀⣀⠸⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⣡⣶⣶⣬⣭⣥⣴⠀⣾⣿⣿⣿⣶⣾⣿⣧⠀⣼⣿⣷⣌⡻⢿⣿⣿⣿⠟⣋⣴⣾⣿⣿⣿⣿⣿⣿⣿⡇⢿⣿⣿⣿⣿⣿⣿⡿⢸⣿⣿⣿⣿⣷⠄⢻⡏⠰⢾⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⢂⣭⣿⣿⣿⣿⣿⠇⠘⠛⠛⢉⣉⣠⣴⣾⣿⣷⣦⣬⣍⣉⣉⣛⣛⣉⠉⣤⣶⣾⣿⣿⣿⣿⣿⣿⡿⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡘⣿⣿⣿⣿⣿⣿⣿⣿⡇⣼⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣇⢸⣿⣿⣿⣿⣿⣿⣿⠁⣿⣿⣿⣿⣿⣿⣿⣿⣿

### Topic 85 (90 docs)

Top words: chinesewe, chinesewe need, need chinesewe, need, need chinese, chinese, chinese need, besoin, chinois, avons besoin

- `1034140_19644 review=4 sentence=sentence_3`: चीनी जरूरतAvem nevoie de chinezi.ພວກເຮົາຕ້ອງຈິດບໍ່ໄດ້আমাদের চীনা দরকারVið þæs gecynd ChinesTreba nam KineskaNous avons besoin de ChinoisVi trenger kinesisk.Vi behöver kineser.Ni bezonas ĉinanSidinga isiShayinaAwak dhéwé butuh KitinéChúng ta cần Trung Quốc.Ci serve cinese.Szükségünk van kínaira.Χρειαζόμαστε κινέζικο.aoānakiaWaxaan u baahan nahay ShiinoBize Çinçe gerek
- `1034140_19644 review=5 sentence=sentence_7`: चीनी जरूरतAvem nevoie de chinezi.ພວກເຮົາຕ້ອງຈິດບໍ່ໄດ້আমাদের চীনা দরকারVið þæs gecynd ChinesTreba nam KineskaNous avons besoin de ChinoisVi trenger kinesisk.Vi behöver kineser.Ni bezonas ĉinanSidinga isiShayinaAwak dhéwé butuh KitinéChúng ta cần Trung Quốc.Ci serve cinese.Szükségünk van kínaira.Χρειαζόμαστε κινέζικο.aoānakiaWaxaan u baahan nahay ShiinoBize Çinçe gerek
- `1144200_80780 review=3 sentence=sentence_1`: 我们需要中文我們需要中文we need Chinese中国语が必要
- `1144200_80780 review=5 sentence=sentence_1`: 我们需要中文我們需要中文we need Chinese中国语が必要
- `1144200_80780 review=7 sentence=sentence_1`: 我们需要中文我們需要中文we need Chinese中国语が必要
- `1227780_1883 review=3 sentence=sentence_1`: 我們需要中文we need Chinese中国语が必要
- `1227780_1883 review=4 sentence=sentence_1`: 我们需要中文we need Chinese中国语が必要
- `1227780_1883 review=5 sentence=sentence_1`: 我們需要中文we need Chinese中国语が必要
- `1227780_1883 review=6 sentence=sentence_1`: 我們需要中文we need Chinese中国语が必要
- `1227780_1883 review=8 sentence=sentence_1`: 我们需要中文we need Chinese中国语が必要

### Topic 86 (90 docs)

Top words: demons, werewolf, named, night shifts, protagonist, kyle, amanda, curse, ritual, job

- `1041920_1898 review=4 sentence=sentence_29`: She’s introduced to the Bandshee, a constellation of freeloaders who spend their time practicing for prospective music gigs in an abandoned, aristocratic mansion referred to as “the big house.”
- `1045720_3045 review=5 sentence=sentence_6`: Our new protagonist, Mina, is this time the unlucky winner of the Coma raffle, dragging her into a corrupt and twisted version of our realm.
- `1065310_9049 review=7 sentence=sentence_1`: 背景设定在架空的19世纪末的西部，在这里牛仔风格的猎手们主要以猎杀吸血鬼相关势力为使命，他们享有较高的社会地位，也有强壮的体魄与高超的技艺。
- `1112890_4627 review=4 sentence=sentence_10`: The other towns inhabitants also have their own dark secrets, some of them are even witches themselves, people deluded into thinking they are owls and a mysterious black-market cat who only speaks in meows.
- `1135810_2645 review=4 sentence=sentence_21`: You’d swear on a stack of mangas this guy was an Elf in some alternate universe, father of the only 3D woman you’d ever get a bodypillow of.
- `1138660_5670 review=3 sentence=sentence_6`: The storyline follows Vasilisa as she confronts demons, bargains with spirits, and unlocks forbidden magic, all while navigating a beautifully haunting world.
- `1138660_5670 review=6 sentence=sentence_15`: Basically, everything morally considered "bad" has a Chort (Demon) behind it, unseen by humans other than Koldun (sorcerers).
- `1166290_6129 review=5 sentence=sentence_12`: Beginning as a trainee Grim Reaper, under strict supervision from our smug boss and his cat, players can choose whether to follow his regime, or to rebel and do as we please.
- `1176470_6007 review=4 sentence=sentence_43`: You play as a secret conspiracy of unelected Deep State bureaucrats who wear countries like skinsuits and play politicians like sock puppets.
- `1179080_6470 review=10 sentence=sentence_18`: You play as a priest who tries to exorcize demons.

### Topic 87 (86 docs)

Top words: hard master, learn hard, easy learn, master, master easy, learn, easy, hard, aprender, fcil

- `1029550_11568 review=3 sentence=sentence_15`: Easy to learn / Hard to master☐
- `1082430_15462 review=6 sentence=sentence_25`: Easy to learn / Hard to master☐
- `1106840_11729 review=7 sentence=sentence_13`: Easy to learn / Hard to master☐
- `1201240_11231 review=6 sentence=sentence_13`: Easy to learn / Hard to master☐
- `1201270_5269 review=4 sentence=sentence_3`: Tidak perlu memakai otak (Mudah)🔲 Gampang, tapi tidak mudah buat jadi jago✅ Perlu pake otak/nalar🔲
- `1253920_14141 review=7 sentence=sentence_8`: ✔️ Fácil de aprender, Difícil de dominar.
- `1272320_7096 review=4 sentence=sentence_22`: Fácil de aprender/difícil de dominar☐ Difícil☐
- `1273710_8900 review=4 sentence=sentence_13`: Easy to learn / Hard to master☐
- `1274290_6350 review=6 sentence=sentence_20`: Easy to learn / Hard to master☐
- `1301720_11577 review=5 sentence=sentence_21`: Easy to learn / Hard to master☑

### Topic 88 (85 docs)

Top words: free, free play, game free, freetoplay, free game, play, game freetoplay, base game, game, f2p

- `1013320_5961 review=6 sentence=sentence_1`: Do you enjoy games with a massive pay wall?
- `1022310_1594 review=3 sentence=sentence_6`: The game should really be Free to Play which would invite more players as the base game doesn't offer enough beyond learning to play.
- `1128810_37138 review=6 sentence=sentence_1`: This is NOT free to play.
- `1145960_5728 review=3 sentence=sentence_8`: Fortunately, a majority of the series is free to play and takes the form of point n click escape room style games.
- `1150080_7309 review=3 sentence=sentence_4`: Fundamentally, an inferior version of the mobile game that is free to play.
- `1170950_12464 review=3 sentence=sentence_26`: Плюсы: - Отсутствие free-to-play.
- `1184140_5300 review=4 sentence=sentence_8`: This is a fun f2p game and you won't want to spend money on it
- `1218250_4574 review=4 sentence=sentence_2`: All pros and cons are considered that the game is developed by a small team and free.
- `1240440_18970 review=3 sentence=sentence_45`: There's not much of a reason not to try this game out given it’s free-to-play.
- `1241510_19675 review=4 sentence=sentence_5`: That's how this game, that should be, imo, free for what it offers, amassed such an overwhelming rating.

### Topic 89 (82 docs)

Top words: mullet madjack, mason lindroth, mullet, madjack, flatout, mason, carnage, limbo, upd, god eater

- `1035120_4085 review=8 sentence=sentence_1`: Сюжет этой части очень тесно переплетается с действующими лицами и событиями из .
- `1045720_3045 review=4 sentence=sentence_18`: Окремо хочу сказати про Єсоль — персонажа з першої частини, яка миттєво полюбилася всім, хто грав.
- `1067540_2680 review=3 sentence=sentence_1`: И подводя итог, хочется отметить следующее, что это невероятная сказка, в которой поднимаются совсем не сказочные вопросы бытия.
- `1105500_5697 review=3 sentence=sentence_9`: Истории каждого главного персонажа становятся более насыщенными по мере прохождения сюжета, и ожидание их встречи в захватывающем финале становится особенно интригующим.
- `1105500_5697 review=3 sentence=sentence_25`: Уже нет банального "принеси-подай", тут вам будут встречаться истории за каждого из персонажей, и зачастую это прописанные, как драматичные так и юморные эпизоды.
- `1105510_5640 review=5 sentence=sentence_17`: Основной сюжет кажется нелепым, хоть и в прошлой игре его нельзя назвать идеальным, но на этот раз он с излишним количеством логических нестыковок и странных объяснений, развивается сам по себе медленно.
- `1105510_5640 review=5 sentence=sentence_18`: Неясно, зачем разработчики затянули сюжет, при этом его искусственно затянули как в третьей части, когда как четвертая уже была с правильным подходом к развитию истории, а постоянные сюжетные повороты в пятой части слишком утомляют игрока к концу игры.
- `1105510_5640 review=5 sentence=sentence_25`: Хоть в третьей части история была спорной, игра не успевала надоедать, диалоги как-никак казались интересными.
- `1105510_5640 review=5 sentence=sentence_27`: Появление Акиямы на экране и его харизма немного разбавили происходящий бред и дали мне сил завершить прохождение.
- `1139940_4015 review=8 sentence=sentence_6`: Умения сценаристов из AtomTeam тоже подтянулось.

### Topic 90 (80 docs)

Top words: simple, easy, game simple, gameplay, gameplay simple, simple game, loop, gameplay loop, mechanics, complex

- `1036890_4908 review=3 sentence=sentence_18`: The gameplay loop is as follows- Lo Wang runs and jumps over floaty land towards a square arena, has to kill a handful of big demons while small ones constantly spawn, he leaves, a cutscene (with poor dubbing) plays and then he runs and jumps again.
- `1057750_6138 review=4 sentence=sentence_3`: The game mechanics are very simple, and clean.
- `1062520_16450 review=4 sentence=sentence_6`: I also enjoy the fast and simple UI of this game.
- `1088790_2820 review=3 sentence=sentence_8`: 每一帖良藥都簡化成遊戲中玩家們可以按按鈕採取的策略。
- `1090630_8516 review=5 sentence=sentence_32`: More complex than any of the other games here due to the need to master 3 characters that are somehow mechanically distinct despite all being Goku.
- `1110050_1747 review=6 sentence=sentence_4`: The gameplay is simple and satisfying.
- `1128810_37138 review=5 sentence=sentence_3`: As RISK games go, this is pretty clean and simple.
- `1147860_3675 review=3 sentence=sentence_7`: Nobody at Mossmouth had to make any of these simpler games to learn.
- `1150080_7309 review=3 sentence=sentence_25`: The game is easy and a breeze to go through.
- `1172450_7999 review=7 sentence=sentence_3`: 游戏的玩法很简单，用一句话来说就是拼图。

### Topic 91 (80 docs)

Top words: bugs, experienced, issues, optimization, crashes, optimized, glitches, havent experienced, encountered, crash

- `1016800_13368 review=3 sentence=sentence_14`: The base building is awesome and as Todd Howard would say "It just works"I've had 0 crashes, and no bugs though I've seen a handful on streams so I know they do exist.
- `1040070_2071 review=3 sentence=sentence_6`: And there were no bugs or problems-- the game ran smoothly and I had no issues with the UI.
- `1056960_2523 review=3 sentence=sentence_4`: 99, 9 % of the time it revived me with zero issues, no matter how far away from it I was, no matter how many and how tough enemies were around me.
- `1089980_30650 review=3 sentence=sentence_2`: Despite the optimization being pretty meh, it didn't stop me from completely 100% the game and obtaining all of the achievements (fairly) in the process.
- `1089980_30650 review=3 sentence=sentence_7`: The optimization is my only huge gripe about the game, but other than that, it is brilliant.
- `1123770_8145 review=4 sentence=sentence_21`: It's even reasonably polished : it's remarkably bug-free, at least in my experience, and for the record even the alpha versions were pretty clean in that regard.*****So, yeah, I have nothing but good things to say about CoDG.
- `1123830_2085 review=3 sentence=sentence_12`: I've enjoyed the game so far and haven't experienced any crashes of freezes.
- `1157220_1663 review=4 sentence=sentence_30`: It's also been rock-solid for me with nary a bug in sight.
- `1173200_2712 review=4 sentence=sentence_11`: Overall the game is very optimized, fast and if you spend a minute to familiarize with the menus, they are lightning fast as well.
- `1173820_3396 review=3 sentence=sentence_11`: I have encountered zero bugs as of The Floating Continent.

### Topic 92 (79 docs)

Top words: metroidvania, metroid, metroidvanias, megaman, metal gear, genre, titulo, hollow knight, metal, msx

- `1062110_1737 review=3 sentence=sentence_16`: Exploration is the bread and butter of metroidvania's and games that are similar to the legend of zelda, with Unsighted getting top marks!
- `1123050_5364 review=4 sentence=sentence_2`: I highly recommend this game, the bosses are the best in a Metroidvania since Hollow Knight so far.
- `1123050_5364 review=4 sentence=sentence_17`: Overall I give this game a 8/10 and HIGHLY recommend this game to any fan of the metroidvania genre.
- `1123050_5364 review=5 sentence=sentence_6`: Thank you for showing me 2021 doesn’t have to be completely full with disappointing games, and for showing me that I might need to stop sleeping on the Metroidvania genre.
- `1191630_1681 review=3 sentence=sentence_5`: It's all here for Shantae fans and fans of metroidvania games.
- `1191630_1681 review=4 sentence=sentence_5`: It's now a proper Metroidvania, the world is a single interconnected map, yet there are still boss dungeons and small challenge caves, just like in other Shantae games.
- `1196630_2868 review=5 sentence=sentence_18`: Travel - I like the Metroid inspired upgrades you get throughout the game (minor spoiler, sorry).
- `1203710_2650 review=4 sentence=sentence_2`: Heavily inspired by Metal Gear 2: Solid Snake from the MSX in character design, general aesthetic and gameplay, UnMetal already has some very witty humour and I am barely an hour in.
- `1203710_2650 review=4 sentence=sentence_4`: My only negative point really is that, if you are familiar with Metal Gear 1 and 2 (again, MSX, not the NES titles) the bosses are pretty much carbon copies, albeit with some humorous additions.
- `1203710_2650 review=6 sentence=sentence_10`: Поэтому если вам нравятся легкие, весёлые, но до мелочей проработанные проекты, то вам стоит обратить своё внимание на "UnMetal", ведь в нем вы найдете не только увлекательный геймплей и хорошую историю, но и сатирическую пародию на первые две части "Metal Gear" Хидэо Кодзимы, а также на весь шпионско-боевиковый колорит последних двадцати лет.

### Topic 93 (78 docs)

Top words: release franchise, franchise release, franchise, release, shrine gain, offer shrine, shrine, gain offer, gain, offer

- `1337520_27372 review=4 sentence=sentence_1`: You offer to the shrine, but gain nothing.
- `1337520_27372 review=4 sentence=sentence_2`: You offer to the shrine, but gain nothing.
- `1337520_27372 review=4 sentence=sentence_3`: You offer to the shrine, but gain nothing.
- `1337520_27372 review=4 sentence=sentence_4`: You offer to the shrine, but gain nothing.
- `1337520_27372 review=4 sentence=sentence_5`: You offer to the shrine, but gain nothing.
- `1337520_27372 review=4 sentence=sentence_6`: You offer to the shrine, but gain nothing.
- `1337520_27372 review=4 sentence=sentence_7`: You offer to the shrine, but gain nothing.
- `1337520_27372 review=6 sentence=sentence_1`: "You offer to the shrine but gain nothing."
- `1337520_27372 review=6 sentence=sentence_2`: "You offer to the shrine but gain nothing."
- `1337520_27372 review=6 sentence=sentence_3`: "You offer to the shrine but gain nothing."

### Topic 94 (77 docs)

Top words: keyboard, player keyboard, escadd, escadd player, xbox, player, pc, emmm, keyboard player, assign keyboard

- `1058830_3923 review=3 sentence=sentence_25`: 再就是比较单一的纯PC玩家了，个人的体验就是采用鼠标的移动来作为模拟打碟的动作，其操作体验比起手柄的小摇杆来更加适配，毕竟只要桌面够大，就可以尽情的刮擦，旋转！
- `1058830_3923 review=3 sentence=sentence_26`: emmm若是用街机的摇杆，可能是个不错的选择！
- `1097350_3724 review=3 sentence=sentence_10`: 游戏的战斗采用的即时制，键盘控制移动，鼠标瞄准射击。
- `1182900_18066 review=3 sentence=sentence_3`: 游戏有中配但听起来比较违和，建议法配或英配游玩，务必使用DS手柄，有非常爽的自适应扳机和触觉反馈。
- `1225570_6759 review=3 sentence=sentence_1`: 的小伙伴们，你们需要：1. 连接并配置手柄。
- `1225570_6759 review=3 sentence=sentence_2`: 推荐使用Xbox 360手柄，其他手柄可能无法识别，此时可以使用Xoutput软件映射键位；2. 进入游戏，在ESC菜单中选择Add another player to keyboard。
- `1225570_6759 review=3 sentence=sentence_6`: 的小伙伴们，建议你们：1. 一人将自己的键盘映射为手柄。
- `1225570_6759 review=3 sentence=sentence_7`: 推荐使用KeyboardSplitterXbox软件，下载地址在页面内下载Keyboard.Splitter.2.2.0.0.zip，解压缩后，配置自己喜欢的键位映射，点击Start启动手柄模拟；2. 映射完成后的玩家在游戏中按下Shift+Tab邀请另一人进行远程同乐，注意给予对方键盘控制权限；3. 另一人进入后，在ESC菜单中选择Add another player to keyboard。
- `1225570_6759 review=4 sentence=sentence_10`: PS:游戏支持本地双人游玩，一个是手柄，一个是键盘。
- `1225570_6759 review=4 sentence=sentence_11`: 进入游戏，到人物控制界面，唤出菜单，（当你有手柄时会出现）里面有个控制选项Add another player on Keyboard 进去后，选择Assign keyboard to player 1或者（2），你尝试一下就可以实现本地双人游玩了。。

### Topic 95 (77 docs)

Top words: oo, icebreakerlust, touchthe, powerliving kanazawarosewater, kanazawarosewater manor88name88s, icebreakerlust powerliving, manor88name88s triangle, manor88name88s, kanazawarosewater, milky touchthe

- `1029550_11568 review=5 sentence=sentence_9`: сливочное масло для смазки готовых блинов 200 гр
- `1097430_6948 review=3 sentence=sentence_1`: 喜欢黄油的可以看看我下面的强烈推荐O(∩_∩)O哈哈~
- `1097430_6948 review=3 sentence=sentence_51`: 牛奶触觉：Milky Touch破冰船：The Icebreaker欲望和权利：Lust and Power住在金泽：Living in Kanazawa—————————————————————玫瑰水庄园：Rosewater Manor名称88的三角形：Name88's Triangle Book欺诈行为：A Deceitful Act淘汰赛大师：Knockut Master Round
- `1144400_27470 review=3 sentence=sentence_1`: 喜欢黄油的可以看看我下面的强烈推荐O(∩_∩)O哈哈~
- `1144400_27470 review=3 sentence=sentence_51`: 牛奶触觉：Milky Touch破冰船：The Icebreaker欲望和权利：Lust and Power住在金泽：Living in Kanazawa—————————————————————玫瑰水庄园：Rosewater Manor名称88的三角形：Name88's Triangle Book欺诈行为：A Deceitful Act淘汰赛大师：Knockut Master Round
- `1145360_56360 review=5 sentence=sentence_8`: HADES加油，真诚的作品永远会得到回报的！
- `1146630_6863 review=5 sentence=sentence_1`: 喜欢黄油的可以看看我下面的强烈推荐O(∩_∩)O哈哈~
- `1146630_6863 review=5 sentence=sentence_51`: 牛奶触觉：Milky Touch破冰船：The Icebreaker欲望和权利：Lust and Power住在金泽：Living in Kanazawa—————————————————————玫瑰水庄园：Rosewater Manor名称88的三角形：Name88's Triangle Book欺诈行为：A Deceitful Act淘汰赛大师：Knockut Master Round
- `1159660_4859 review=5 sentence=sentence_6`: 玄米茶，日韩风味绿茶饮品，感兴趣可自行搜索红场，(俄文 красная площадь 英文 Red Square)位于俄罗斯首都莫斯科市中心，临莫斯科河，是莫斯科最古老的广场 ，是重大历史事件的见证场所。
- `1188930_11230 review=4 sentence=sentence_2`: 双子：威士忌，小提琴，推理小说，杂志阿扎尔：咖啡，手枪，少年漫画蕾琳：艾草，浪漫dvd，清酒，科幻小说约翰：浪漫dvd 牛奶 平底锅 伏特加伊利亚：钓鱼竿 相机 伏特加 清酒）

### Topic 96 (77 docs)

Top words: infinity, infinity price, long infinity, average long, price, average, calculadora, com pilha, uma calculadora, pilha

- `1029550_11568 review=3 sentence=sentence_28`: To infinity and beyond---{ 🪙Price🪙 }---☑
- `1082430_15462 review=6 sentence=sentence_34`: To infinity and beyond---{Price}---☐
- `1145960_5728 review=4 sentence=sentence_16`: Sonsuzluk ve Ötesine---{Fiyat}---☐
- `1158940_9072 review=7 sentence=sentence_18`: Çöpten bulunan bilgisayar🔲 Tost makinesi☑️ Ortalama🔲
- `1180320_61045 review=12 sentence=sentence_1`: 哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈.......
- `1181830_2282 review=3 sentence=sentence_9`: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- `1197570_4349 review=3 sentence=sentence_6`: ——————————————————————————————————————————————————————
- `1262350_14960 review=3 sentence=sentence_4`: ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎ ‎
- `1262560_20227 review=4 sentence=sentence_4`: Uma calculadora com 1 pilha✅
- `1272320_7096 review=4 sentence=sentence_24`: European Extreme⏳===DURACION===

### Topic 97 (75 docs)

Top words: dry instead, watch paint, paint dry, dry, paint, watch, midnight paradise, instead watch, midnight, paradise

- `1029550_11568 review=3 sentence=sentence_5`: Watch paint dry instead☐
- `1083880_2779 review=4 sentence=sentence_30`: 夜中に城に侵入し（Sleep Powderでガードを眠らせる、特定の姫の力を使う）ガードに見つからずに姫の寝室に辿り着いて姫に話しかけると十分な好感度があれば致す事が出来る
- `1097430_6948 review=3 sentence=sentence_21`: 午夜天堂：Midnight Paradise
- `1106840_11729 review=7 sentence=sentence_7`: Watch paint dry instead☐
- `1119730_16960 review=4 sentence=sentence_2`: podia ter NPC andando, vizinhos , ter a opção de como excluir as latas de tinta, ter como plantar tomates, estufas
- `1144400_27470 review=3 sentence=sentence_21`: 午夜天堂：Midnight Paradise
- `1146630_6863 review=5 sentence=sentence_21`: 午夜天堂：Midnight Paradise
- `1201240_11231 review=6 sentence=sentence_7`: Watch paint dry instead☐
- `1213740_18053 review=3 sentence=sentence_19`: 午夜天堂：Midnight Paradise
- `1216320_6022 review=7 sentence=sentence_7`: Watch paint dry instead☐

### Topic 98 (74 docs)

Top words: trailer, trailers, sequel, series, fan series, srie, franchise, game look, series id, waiting years

- `1030210_11120 review=3 sentence=sentence_4`: And man, what a disappointment it was to see how the series devolved with this entry.
- `1030210_11120 review=3 sentence=sentence_13`: It’s a long and bumpy road though and it’s a shame, because instead of reinvigorating the love for the series, brought it a step closer towards falling from grace.
- `1063660_11863 review=4 sentence=sentence_3`: oozes atmosphere with its presentation, being a substantial upgrade compared to and even compared to its own trailers in the years since announcement.
- `1084160_11858 review=3 sentence=sentence_3`: I think I can say that I'm a pretty big fan of the series.
- `1088710_8240 review=4 sentence=sentence_4`: Seri herkese göre değil öncelikle.
- `1088710_8240 review=5 sentence=sentence_6`: Mantém o alto padrão de qualidade da série e vai que vai!
- `113200_6688 review=4 sentence=sentence_8`: Just get the sequel.
- `1145350_50050 review=4 sentence=sentence_2`: Always the first series remains in the mind and hearts.
- `1180380_1256 review=5 sentence=sentence_4`: I've followed the ENTIRE stalker franchise since its release and enjoy this quite a bit given its obvious unfinished state.
- `1214650_3444 review=6 sentence=sentence_8`: Os trailers enganam demais!

### Topic 99 (74 docs)

Top words: campaign, campanha, campaigns, imperialism, kampagne, just campaign, campaign does, endgame content, verschiedene, scripted

- `1069650_1910 review=3 sentence=sentence_7`: 또 각종 버그라든가 아직 캠페인 길이가 짧다든가 하는 문제점이 있어요.
- `1072040_3577 review=4 sentence=sentence_7`: Исторические и неисторические ветки кампании.
- `1072040_3577 review=5 sentence=sentence_12`: ebenso wenig verschiedene Siegmöglichkeiten wie früher (taktischer, normaler, glorreicher Sieg) die verschiedene Wege im weiteren Verlauf der Kampagne geben.
- `1072040_3577 review=5 sentence=sentence_19`: Die Kampagne ist recht linear. -Auch wenn man einmal zwischen Afrika und Sovietunion wählen darf.
- `107410_23399 review=10 sentence=sentence_2`: The main campaign is interesting, and there is a definite learning curve.
- `1101190_4659 review=3 sentence=sentence_9`: Chromatic (and Trendy) have an excellent track record of making an enjoyable campaign.
- `1119700_2053 review=3 sentence=sentence_9`: Vous pensiez que la campagne était difficile ?
- `1119700_2053 review=4 sentence=sentence_8`: Also, the campaign is not very fun.
- `1119700_2053 review=4 sentence=sentence_10`: The "Campaign" does teach some later level techniques, but the regular game also has guidance on how to move and generally what to do.
- `1119700_2053 review=4 sentence=sentence_11`: If the campaign gets frustrating, skip it entirely and come back to it later.

### Topic 100 (73 docs)

Top words: developers, dev, developer, devs, developed, person, game, game developed, studios, small

- `1000360_6765 review=3 sentence=sentence_26`: The developer has shown a large amount of faith and interest in his own game, which, again, is such a breath of fresh air after having dealt with the AAA companies that own the FGC circuit as of late.
- `1012790_10355 review=3 sentence=sentence_1`: I could endlessly attest to the sheer quality of this game, but instead I will tell you how much the developers care about it's quality to prove it.
- `1040200_11371 review=5 sentence=sentence_23`: This feels like one of the RARE times the devs have played their own game and are fans of the genre.
- `1106840_11729 review=5 sentence=sentence_24`: The developer is hard at work to make this game fantastic, I urge you to give it a try.
- `1130410_3627 review=8 sentence=sentence_13`: Kudos to the dev team for making such a good game!
- `1136370_4624 review=5 sentence=sentence_9`: Absolutely wholesome devs and game!
- `1140270_5797 review=4 sentence=sentence_42`: Final one, and this is less about the game, and more about the Devs.
- `1148590_7255 review=6 sentence=sentence_3`: As a matter of fact, it is the only DOOM game that was not developed by ID Software but was made under their supervision by Midway Studios.
- `1153640_3722 review=4 sentence=sentence_10`: Also really nice for a dev studio to put out a finished game.
- `1157220_1663 review=4 sentence=sentence_29`: But unlike most games that are thrown out the door and abandoned, the devs have made Neb into a really great title.

### Topic 101 (72 docs)

Top words: level design, design, level, levels, designed, design good, game level, games level, design bit, great level

- `1000410_2404 review=3 sentence=sentence_11`: -Levels are geometrically complex for the Quake engine and impressive in that regard, however...-While they are complex, they're visually uninteresting.
- `1000410_2404 review=3 sentence=sentence_12`: Most levels are plagued by bland color palettes and setpieces, which is further exacerbated by the fact that each level in an episode looks almost identical to the other in design.
- `1069650_1910 review=3 sentence=sentence_18`: 다만 레벨 디자인이 살짝 애매해요;;
- `1087760_2041 review=3 sentence=sentence_14`: While each area contains some small side-areas with extra crafting materials or the odd “out of the beaten path” research article, the game’s level design is largely linear, subdivided in zones you can return to at any time via a convenient fast-travel system unlocked not too far into the story.
- `1130410_3627 review=8 sentence=sentence_7`: Most of the levels are well designed, with a few exceptions.
- `1150440_11610 review=3 sentence=sentence_9`: The RTS portions of the game are very well done, with level design standing out.
- `1150760_5674 review=3 sentence=sentence_6`: The levels are well made, and there's plenty of secrets to find, but that's all the content there is.
- `1150950_2354 review=4 sentence=sentence_14`: ただ本作が優れていると感じるのは、そうした気付きにくい解法も進行上で必ず気付くようなレベルデザインとなっている点です。
- `1191630_1681 review=5 sentence=sentence_12`: The level-based structure from the last game is now gone and the interconnected world from the early games is back and it's probably the best one in the series.
- `1213700_4611 review=4 sentence=sentence_33`: Несовершенство дизайна уровней.2.

### Topic 102 (72 docs)

Top words: crashes, crash, crashing, game crashes, game crash, crashed, crash game, random, screen, constantly crashes

- `1015500_833 review=6 sentence=sentence_3`: It constantly crashes every hour or so and not only that 60% of the content in this game including movesets and attires for custom wrestlers are locked behind a pay wall.
- `1015500_833 review=7 sentence=sentence_1`: Really love the all the creation options, from belts to arenas, and all the wrestlers, but the game crashes and if you try to start a custom universe by adding aew or indie venues the game will not save.
- `1015500_833 review=9 sentence=sentence_12`: I was annoyed by this because, while it wasn't like I had never had experienced a game crash on 2K19, that game at least managed to get through the regular weekly shows without a problem.
- `1015500_833 review=9 sentence=sentence_18`: Glad that everything was working right now, I tried to load into the second match aaaaand the game crashed.
- `1017900_1112 review=3 sentence=sentence_5`: but i have completely random freezing and crashing issues that make it incredibly frustrating to play.
- `1017900_1112 review=3 sentence=sentence_11`: i can be doing nothing at it freezes.
- `1041720_6700 review=5 sentence=sentence_6`: Even after months, the game is susceptible to crashes rendering the whole experience as unenjoyable.
- `1069640_4384 review=3 sentence=sentence_9`: After capturing more than 50% of the map, the game started to crash every 30 minutes.
- `1196090_2357 review=5 sentence=sentence_4`: My biggest problem is that the game crashes every time you sprint into a barrier.
- `1222730_26432 review=5 sentence=sentence_6`: In campaign mode, I notice that if you're playing in VR, at certain random point, the game will hard lock and you won't be able to go any farther.

### Topic 103 (71 docs)

Top words: camera, angle, view, camera angle, camra, screws, cameras, fixed camera, angles, camra est

- `1016790_1893 review=3 sentence=sentence_10`: Fixed camera angles lead to not being able to see inside some of the rooms without just running into them• The melee enemies throw the cover and shoot combat out the window• Lacks polish and refinement making it feel like an early access game•
- `1016790_1893 review=3 sentence=sentence_26`: Unfortunately, this becomes a problem with the fixed camera angles, if you're running towards the camera going into a room you will have no idea of where the enemies might be or where the lanterns are.
- `1049410_23830 review=3 sentence=sentence_6`: Sua camera se movimenta mto, alguns pontos são quase impossiveis para jogar
- `1055540_1264 review=3 sentence=sentence_11`: The only problem I have with the game is the finnicky camera, adjusting the angle can be a pain in certain areas.
- `1055540_1264 review=6 sentence=sentence_10`: There is almost nothing to hate about this game expect the camera angles could have been a little more lenient.
- `1101790_2018 review=3 sentence=sentence_19`: The isometric view makes it difficult to shoot accurately, and I almost wish that the game somehow was turn-based when it comes to shooting
- `1167140_1593 review=4 sentence=sentence_35`: Top-down camera: more freedom of movement.
- `1190460_8752 review=4 sentence=sentence_48`: Oh, and the camera doesn't auto-center behind the vehicle when turning.
- `1190970_8810 review=3 sentence=sentence_11`: Why give us the option to rotate the camera and change our field of view if the hammer doesn't auto-orientate with it?
- `1194630_8348 review=5 sentence=sentence_13`: Keeping the camera dynamic, unlike in Man of Medan, is a visible improvement from past experiences.

### Topic 104 (68 docs)

Top words: platforming, platformer, platformers, sections, platforming sections, puzzles, challenging, platformer sections, good platformer, fights

- `1062110_1737 review=3 sentence=sentence_32`: Platforming can be a little tricky at first, since you can't slow your forward momentum too well and overshoot platforms if you do a running jump.
- `1116580_2386 review=5 sentence=sentence_1`: is a beautiful puzzle platformer with stellar art design, excellent soundtrack that mellows you out, many creative puzzles and well executed gameplay mechanics.
- `1123050_5364 review=5 sentence=sentence_5`: From combat to platforming and everything in between, this game really resonates with me.
- `1130410_3627 review=8 sentence=sentence_8`: These exceptions all focus on precise platforming which doesn't quite work with the mechanics.
- `1186640_4980 review=4 sentence=sentence_1`: is a spooky 3D platformer with a phenomenal story, great gameplay, some fantastic visuals, and a comfortable soundtrack.
- `1186640_4980 review=4 sentence=sentence_27`: Pumpkin Jack is nearly perfect in all his aspects: the platforming, puzzles, and the "chases" sequences are magnific.
- `1186640_4980 review=4 sentence=sentence_31`: However, coming back to Pumpkin Jack, the platforming parts of the stages are good, and some of them have interesting and fun puzzles that obligate the player to think.
- `1220150_2723 review=4 sentence=sentence_5`: The game seems to focus more on platforming (- pretty good; it's smooth, has a variety of styles, with precision-platforming challenges to boot) than combat (- just okay; it's the same re-skinned enemies and really not that much to fight... few, mostly simple bosses).
- `1220150_2723 review=6 sentence=sentence_8`: - Challenging and unique platformer sections- Power-ups to make the platformer sections easier- Widely explorable areas that hide secrets and easter eggs- Enemies can disrupt the platforming sections- Combat relies too much on platforming controls- Dashing to flying enemies tend to not work
- `1220150_2723 review=6 sentence=sentence_26`: The only thing that you need to watch out for is the existence of s, a closed-off platformer section that can give you an extra heart if you manage to complete it.

### Topic 105 (67 docs)

Top words: madrid, und ich, spieler, und, mick, mannschaft, meine mannschaft, der, nicht weg, caej

- `1196470_1418 review=4 sentence=sentence_9`: es gibt Spielstände wo meine Mannschaft 30 Spieler hat und ich 12 auf der Transfer-Liste habe und ich keinen verkauft oder ausgeliehen bekomme seit 2 Saison- Speichern nur am Anfang der Woche möglich- Jugendspieler verlangen zu viel Handgeld, selbst in der 4ten Liga, und über die Ausstiegsklausel die viele Spieler wollen und man sie nicht weg bekommt, reden wir lieber nicht. - Transfermarkt ist komplett Rip.
- `1263850_23896 review=3 sentence=sentence_4`: 但MC位置 远射20 和远射1 区别不大，该进还是进。
- `1263850_23896 review=3 sentence=sentence_5`: DC位置，（设置成非出球后卫的职责）传球20和传球1，都会频繁出球。
- `1313860_25945 review=3 sentence=sentence_20`: You guys remind me of Hazard's glorious impact at Real Madrid.
- `1569040_21403 review=3 sentence=sentence_1`: 我年纪轻轻就考取了洲际教练证书，正是志得意满之时，四处求职，最后有一个英乙的小球队找到我，说可以让我来做新的主教练，但是他们今年的期望是升级。
- `1569040_21403 review=3 sentence=sentence_3`: 我上任之后，立刻花钱引入了几个实力足够在英甲联赛踢球的失意者，我说，给你们主力的位置，我们今年踢一年英乙，明年就升级英甲，来我的队。
- `1569040_21403 review=3 sentence=sentence_5`: 最终拥有英甲优秀球队实力的我们在英乙一路高歌猛进，以毫无悬念的大优势获得了当年的英乙冠军，成功进入英甲联赛，队内我十分看好的射手新星也荣膺英乙射手王。
- `1569040_21403 review=3 sentence=sentence_6`: 赛季结束后，利物浦发邮件联系到我，说想要签下我们队里的头号射手。
- `1569040_21403 review=3 sentence=sentence_10`: 他说：我要为了我的职业生涯着想，你要是为了我好，就应该让我去更大的球会发展。
- `1569040_21403 review=3 sentence=sentence_12`: 他说：利物浦给我开的只是青年球员合同，但我也要去。

### Topic 106 (65 docs)

Top words: environment, environments, atmosphere, beautiful, scenery, etkileim, atmosphere just, ne ok, vistas, stunning

- `1040200_11371 review=5 sentence=sentence_5`: VERY lived in environments.
- `1055540_1264 review=3 sentence=sentence_2`: Hawk Peak can feel both small and quaint but sprawling with plenty to explore at the same time!
- `1060230_1945 review=4 sentence=sentence_8`: Despite this, the scenery is stunning.
- `1060230_1945 review=4 sentence=sentence_9`: You can bring the camera down to the eye level of your sapiens and soak in the view they have of the mountains and rivers.
- `1082680_1072 review=3 sentence=sentence_17`: Das erlaubt es uns, einen näheren Blick auf unsere Umgebung zu werfen.
- `1082680_1072 review=3 sentence=sentence_43`: Die Umgebung ist recht steril und wie gesagt.
- `1159690_5248 review=5 sentence=sentence_5`: seriously, the environmental design is really wonderful and whimsical and cruising through it all on the floating rails is such a pleasant visual.
- `1194630_8348 review=5 sentence=sentence_14`: This allows the player to appreciate the beautiful environment considerably more.
- `1196090_2357 review=4 sentence=sentence_17`: Each zone was quite beautiful and some of the views were amazing.
- `1211020_23610 review=3 sentence=sentence_20`: 〽️Çevre ile etkileşim 10 numara 5 yıldız.

### Topic 107 (65 docs)

Top words: p5, djmax, psp, ps5, reborn, ps3, enhanced, persona, ps, version

- `1173780_2330 review=3 sentence=sentence_28`: 顺势我又把psp版+psp魂之重生重新通关了一遍，深刻体会到，什么才叫真正的良心重制。
- `1173780_2330 review=3 sentence=sentence_30`: 推荐大家都去找个模拟器玩psp版，感受一下SE在15年前的诚意。
- `1295510_9188 review=3 sentence=sentence_19`: 不仅把PS4的3D和3DS的2D画面都加入了，还加入了全语音这种DQ系列从来没有过的东西，另外诚意十足地加入了全同伴结婚的选项（可耻渣男哑巴伊莱文抛弃青梅竹马村姑），此外更是给保健室老师露洁多加了套色气的衣服。
- `1328670_19939 review=3 sentence=sentence_10`: PPS 吧宝代码应该可以直接用原版游戏的，问题不大。
- `1337520_27372 review=7 sentence=sentence_2`: Под впечатлением от недавно проданной PSP я покупаю PS Vita и при просмотре библиотеки натыкаюсь на смутно знакомое название Risk of Rain.
- `1382330_15081 review=4 sentence=sentence_2`: pralo para que saquen el Persona 5 Royal en PC⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀ ⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
- `1382330_15081 review=5 sentence=sentence_2`: 这条推特不禁令很多人遐想连篇，很多人都认为这是在暗示女神异闻录5即将登录pc平台，让我们拭目以待吧！
- `1382330_15081 review=7 sentence=sentence_1`: 事实上，P系列在平台上的割裂使得Steam版本如今登录的P5S并未有太多的购买价值，P5正传的独占使得新购买P5S的PC玩家（以及一些NS玩家）一头雾水，不明就里。
- `1382330_15081 review=7 sentence=sentence_3`: 索尼出于自己的机器售卖利益而独占作品使得其跨平台的衍生作玩家无法享受任何游玩体验，P系列也不应当只被“有PS4的玩家”所喜爱，如今很多没玩过原作的玩家不喜欢P5S而给差评，就是独占策略带给我们的最坏恶果。。
- `1382330_15081 review=7 sentence=sentence_4`: 但与此同时，这款登陆了Steam端的第一部P5系列作品仍然值得所有人游玩。

### Topic 108 (65 docs)

Top words: winning son, winning, psychopath, psychopath psychopath, son winning, son, ya winning, ya, sir, stoic

- `1129540_7403 review=3 sentence=sentence_1`: ⢸⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⡷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠈⠑⢦⡀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢Are you winning son?⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀...
- `1202690_5132 review=3 sentence=sentence_2`: Are you winning son?⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⡄⠀⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⢠⠃⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⢸⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀ ⠷
- `1202690_5132 review=4 sentence=sentence_1`: ⢸⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⡷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠢⣀ARE YA WINNING, SON?⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠈⠑⢦⡀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⡄⠀⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀...
- `1202690_5132 review=5 sentence=sentence_2`: Are you winning son?⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀ ⠀⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⡄⠀⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁ ⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀ ⠀ ⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀ ⠀ ⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀ ⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉ ⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀ ⠀ ⠀⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⢸⢸⠀⠀⠀⠀⢠⠃⠀⠀⡇⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⢸⢸⠀⠀⠀⠀⢸⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠷ ⢸
- `1202690_5132 review=6 sentence=sentence_2`: Are you winning son?
- `1202690_5132 review=7 sentence=sentence_1`: ⢸⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⡷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠢⣀⠀⠀⠀⠀ARE YOU WINNING SON?⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠈⠑⢦⡀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⠀⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀...
- `1202690_5132 review=11 sentence=sentence_2`: Are you winning son?⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⡄⠀⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⢠⠃⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀ ⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⢸⢸⠀⠀⠀⠀⢸⠀⠀⠀⠁⠀⠀⠀⠀⠀⠀⠀ ⠷
- `1238810_76940 review=7 sentence=sentence_11`: Είναι λυπηρό να έχουμε μια μητέρα χωρίς προσθήκη, αλλά δεν ξέρουμε ποιος είναι ο πατέρας του.
- `1268110_3755 review=6 sentence=sentence_1`: ⢸⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⡷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⡇⠢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠈⠑⢦⡀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀⡇⠀⠀⠀ARE⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀⡇⠀⠀⠀⠀⠀⠀⠀YA⠀⠀⠑⠢⡄⠀⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀WINNING⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸ SON?⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀...
- `1271710_7231 review=3 sentence=sentence_1`: ⢸⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⠉⡷⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠢⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠈⠑⢦⡀⠀⠀⠀⠀⠀Are you winning son?⢸⠀⠀⠀⠀⢀⠖⠒⠒⠒⢤⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠙⢦⡀⠀⠀⠀⠀⢸⠀⠀⣀⢤⣼⣀⡠⠤⠤⠼⠤⡄⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠙⢄⠀⠀⠀⠀⢸⠀⠀⠑⡤⠤⡒⠒⠒⡊⠙⡏⠀⢀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠑⠢⡄⠀⢸⠀⠀⠀⠇⠀⣀⣀⣀⣀⢀⠧⠟⠁⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⢸⠀⠀⠀⠸⣀⠀⠀⠈⢉⠟⠓⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠈⢱⡖⠋⠁⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⣠⢺⠧⢄⣀⠀⠀⣀⣀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⣠⠃⢸⠀⠀⠈⠉⡽⠿⠯⡆⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⣰⠁⠀⢸⠀⠀⠀⠀⠉⠉⠉⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠣⠀⠀⢸⢄⠀⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⢸⠀⢇⠀⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⢸⠀⠀⠀⠀⠀⡌⠀⠈⡆⠀⠀⠀⠀⠀⠀⠀⡇⠀⠀⠀...

### Topic 109 (65 docs)

Top words: total war, total, war, rome, rts, war games, shogun, warhammer, strategy, war game

- `1058650_7389 review=4 sentence=sentence_1`: I'm more of a War of Rights and Hold Fast kinda guy.
- `1134100_3473 review=3 sentence=sentence_12`: They really should lean harder into the RTS aspects, that it's similar to Age of Empires.
- `1134100_3473 review=3 sentence=sentence_20`: Again, good game, good RTS, really enjoyable, just absolutely not a city builder.
- `1134100_3473 review=6 sentence=sentence_4`: I have been an avid RTS player my whole life, and over the last five years, I have not found one game that brought back the nostalgic feeling of some of my favorites: Lord of the Rings Battle for Middle Earth 1 & 2, Dawn of War, Age of Empires, and Command and Conquer.
- `1142710_81157 review=3 sentence=sentence_2`: Make no mistake, they fully intend to try this again as proven by years of experience Total War: Three Kingdowms and Total War: Rome 2.-People here often forget the dual nature of the main current sticking point.
- `1157390_6381 review=3 sentence=sentence_4`: Was lukewarm on Expeditions: Rome.
- `1167140_1593 review=4 sentence=sentence_17`: this makes you think about how you want to bring on a mission with you.-----------------------------This game is different from ''This war of mine''
- `1167140_1593 review=4 sentence=sentence_41`: More good things you should find for yourself.-----------------This game is different from ''This war of mine''
- `1330460_4673 review=5 sentence=sentence_1`: Осовремененная RTS-защита базы с интересными идеями.
- `1330460_4673 review=5 sentence=sentence_12`: Очень понравилось, что, в отличии от традиционных RTS, ресурсы в источниках не заканчиваются, что позволяет плотнее сосредоточиться собственно на игре.

### Topic 110 (63 docs)

Top words: tutorial, tutorials, das tutorial, das, habe ich, los tutoriales, content doing, tutorials tutorial, tutorial single, tutoriales

- `1013320_5961 review=3 sentence=sentence_3`: The tutorial shoved me in the in-game shop.
- `1013320_5961 review=3 sentence=sentence_20`: And that the tutorial literally does not allow you to say "no thanks I'll wait".
- `1063420_3769 review=6 sentence=sentence_12`: Skipping the tutorial is a tutorial for instant regret - how tf does this crap work?!
- `1069640_4384 review=5 sentence=sentence_13`: Die Steuerung ist teilweise etwas ungewöhnlich, Tutorial mitlesen sei hiermit dringendst ans Herz gelegt, das ist aber auch überschaubar.
- `1069650_1910 review=3 sentence=sentence_41`: 이번 작엔 튜토리얼도 매우 충실하니, 보고 잘 따라하시면 어려울 건 하나도 없어요.
- `1154840_1353 review=4 sentence=sentence_6`: Here’s a link to ’s tutorial series, which is helpfully divided into basic rules and in-depth analysis.
- `1173200_2712 review=4 sentence=sentence_13`: Chapter 1 being a tutorial and isn't replayable.
- `1173220_2309 review=3 sentence=sentence_31`: The tutorials are good, but there's no way to re-access them once you make them disappear.
- `1173220_2309 review=3 sentence=sentence_32`: There should be an option in the pause menu to re-read the tutorials.-
- `1180380_1256 review=9 sentence=sentence_2`: After trudging through the tutorial zone, practically having to look up a guide for almost every quest, I finally made it to Lubech.

### Topic 111 (62 docs)

Top words: graphics forget, forget reality, reality, reality graphics, forget, graphics, voc esquece, esquece, esquece que, que realidade

- `1029550_11568 review=3 sentence=sentence_2`: ig---{ 📺Graphics📺 }---☐ You forget what reality is☐ Beautiful☐ Good☑ Decent☐ Bad☐
- `1063660_11863 review=6 sentence=sentence_1`: ---{ Gráficos }---☐ Você esquece o que é a realidade☑
- `1106840_11729 review=7 sentence=sentence_1`: ---{ Graphics }---☐ You forget what reality is☐
- `1182620_11227 review=4 sentence=sentence_35`: 或许读到这里xdm会觉得我偏题了，这里明明是在讲“现实度”这个概念啊？
- `1201240_11231 review=6 sentence=sentence_1`: ---{ Graphics }---☐ You forget what reality is☐
- `1201540_3395 review=6 sentence=sentence_1`: =={ Графика }==☐ Забудешь, что такое реальность☐
- `1216320_6022 review=7 sentence=sentence_1`: ---{ Graphics }---☐ You forget what reality is☐
- `1273710_8900 review=4 sentence=sentence_1`: ---{ Graphics }---☐ You forget what reality is☐
- `1274290_6350 review=6 sentence=sentence_1`: ---{Graphics}---☐ You forget what reality is☐
- `1280770_16276 review=7 sentence=sentence_1`: ---{ Graphics }---☐ You forget what reality is☐

### Topic 112 (61 docs)

Top words: little nightmares, northern journey, northern, nightmares, journey, npc, little, games

- `1030210_11120 review=5 sentence=sentence_11`: Разработчики каким-то абсолютно невероятным образом к х*ям ломают то, что работало раньше и начинают медленно чинить то, что уже было чинено и (казалось бы!) "работает - не трогай", но нет.
- `1045720_3045 review=4 sentence=sentence_16`: І ці замітки показують, що розробники мають далекоглядні плани на світ Коми, і це дуже класно.
- `1069640_4384 review=4 sentence=sentence_2`: Хлопці невеликою групою розробників проробили грандіозну роботу і втілили в життя оригінальну задумку протистояння корінного населення Північної Америки - колонізаторам.
- `1139940_4015 review=8 sentence=sentence_12`: Со стороны разработчиков идут какие-то тухлые попытки натянуть сову на глобус путем добавления ненужных фич в систему, которая и так не очень-то работает.
- `1170950_12464 review=3 sentence=sentence_11`: Но стоит признать, разработчики очень активно патчат игру и процесс разработки чувствуется даже поиграв недельку.
- `1203710_2650 review=6 sentence=sentence_3`: Но абсолютно точно к этому перечню выдающихся людей можно с легкостью добавить испанского инди-разработчика Франциско Теллеса де Менезеса.
- `1225560_4765 review=6 sentence=sentence_9`: 👉 Радує те, що розробники просто розповідають історію життя через візуальні образи.
- `1231990_1077 review=6 sentence=sentence_1`: Разработчики в 2020-м году: …Разработчики в 2021-м году: …Разработчики в конце февраля 2022-го года: …Разработчики в 2023-м году: …Разработчики в 2024-м году (полтора месяца до релиза):
- `1232580_2000 review=3 sentence=sentence_2`: Он в игре есть, по крайней мере, об этом заявляют разработчики, но вот разобраться в его хитросплетениях пока не так-то легко.
- `1232580_2000 review=3 sentence=sentence_47`: А вот визуальная часть - это то, за что разработчиков хочется искренне поблагодарить.

### Topic 113 (59 docs)

Top words: replayability, replay, replay value, value, replayable, challenges, good replayability, difficulty, different, add

- `105600_54613 review=12 sentence=sentence_8`: It's that replayability, and the dev team's eagerness to continue which really makes this game.
- `1058830_3923 review=4 sentence=sentence_13`: Lastly, the various difficulties and upcoming challenge modes provide (and will provide) an good amount of replayability.
- `1069650_1910 review=4 sentence=sentence_6`: es ermöglicht es zwar dass man mal mitten in dem spiel wieder mit kleineren schiffen spielen kann wenn man das möchte.
- `1108590_3491 review=3 sentence=sentence_52`: Recomendo àqueles que gostam do fator replay (que Eldest Souls faz muito bem, visto que a dificuldade realmente aumenta e não é apenas a clássica palhaçada de dar mais vida e dano pros inimigos, enquanto todo o resto continua a mesma merda).
- `1114220_3476 review=3 sentence=sentence_3`: Has excellent replay value because of the variable classes you can play.
- `1116580_2386 review=4 sentence=sentence_13`: パズルもアクションも適度な難度でありつつ右トリガーを押すことにより時間を巻き戻すことが可能なのでリプレイのストレスがありません。
- `1116580_2386 review=4 sentence=sentence_14`: 時間を巻き戻す機能はあくまでプレイアビリティを高めるためのものでこれを利用した謎解きはありませんがこれがあるのとないのでは印象が変わったと思います。
- `1150950_2354 review=4 sentence=sentence_16`: 実績にもなっているレリックチャレンジはまさにそうしたノウハウを活かさないと獲得できないようになっており、リプレイを促す要素として機能しています。
- `1189630_30108 review=3 sentence=sentence_6`: This is in of itself quite unique and enjoyable, and does add considerable replay value interms of trying to do different builds or pursue different tasks.
- `1191120_3113 review=5 sentence=sentence_15`: The game does have enough new content to offer in each new playthrough to keep you interested, even if just for the sake of collecting stuff.

### Topic 114 (58 docs)

Top words: janken, isekai janken, simulator fusion, isekai, janken hero, fusion, hero simulator, fusion isekai, hero, simulator

- `1094520_22432 review=5 sentence=sentence_2`: 异世界猜拳勇者：Isekai Janken Hero命运模拟:2024：Orgasm Simulator
- `1094520_22432 review=5 sentence=sentence_3`: 2024命运模拟器2023：ORGASM SIMULATOR 2023召唤与合体：Summon and fusion
- `1097430_6948 review=3 sentence=sentence_2`: 异世界猜拳勇者：Isekai Janken Hero
- `1097430_6948 review=3 sentence=sentence_4`: 2024命运模拟器2023：ORGASM SIMULATOR 2023召唤与合体：Summon and fusion
- `1144400_27470 review=3 sentence=sentence_2`: 异世界猜拳勇者：Isekai Janken Hero
- `1144400_27470 review=3 sentence=sentence_4`: 2024命运模拟器2023：ORGASM SIMULATOR 2023召唤与合体：Summon and fusion
- `1146630_6863 review=5 sentence=sentence_2`: 异世界猜拳勇者：Isekai Janken Hero
- `1146630_6863 review=5 sentence=sentence_4`: 2024命运模拟器2023：ORGASM SIMULATOR 2023召唤与合体：Summon and fusion
- `1213740_18053 review=3 sentence=sentence_3`: 异世界猜拳勇者：Isekai Janken Hero命运模拟:2024：Orgasm Simulator 2024命运模拟器2023：ORGASM SIMULATOR 2023召唤与合体：Summon and fusion
- `1289310_122135 review=7 sentence=sentence_1`: 黄油给你，点赞给我，富哥可以给个奖赏，想买涩涩头像，感谢感谢异世界猜拳勇者：Isekai Janken Hero

### Topic 115 (58 docs)

Top words: cruub, cruub cruub, king cruub, cruub king, monke, king, wren, god, jr, ya

- `105600_54613 review=5 sentence=sentence_1`: ⠀⠀⠀⡯⡯⡾⠝⠘⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢊⠘⡮⣣⠪⠢⡑⡌ ⠀⠀⠀⠟⠝⠈⠀⠀⠀⠡⠀⠠⢈⠠⢐⢠⢂⢔⣐⢄⡂⢔⠀⡁⢉⠸⢨⢑⠕⡌ ⠀⠀⡀⠁⠀⠀⠀⡀⢂⠡⠈⡔⣕⢮⣳⢯⣿⣻⣟⣯⣯⢷⣫⣆⡂⠀⠀⢐⠑⡌ ⢀⠠⠐⠈⠀⢀⢂⠢⡂⠕⡁⣝⢮⣳⢽⡽⣾⣻⣿⣯⡯⣟⣞⢾⢜⢆⠀⡀⠀⠪ ⣬⠂⠀⠀⢀⢂⢪⠨⢂⠥⣺⡪⣗⢗⣽⢽⡯⣿⣽⣷⢿⡽⡾⡽⣝⢎⠀⠀⠀⢡ ⣿⠀⠀⠀⢂⠢⢂⢥⢱⡹⣪⢞⡵⣻⡪⡯⡯⣟⡾⣿⣻⡽⣯⡻⣪⠧⠑⠀⠁⢐ ⣿⠀⠀⠀⠢⢑⠠⠑⠕⡝⡎⡗⡝⡎⣞⢽⡹⣕⢯⢻⠹⡹⢚⠝⡷⡽⡨⠀⠀⢔ ⣿⡯⠀⢈⠈⢄⠂⠂⠐⠀⠌⠠⢑⠱⡱⡱⡑⢔⠁⠀⡀⠐⠐⠐⡡⡹⣪⠀⠀⢘ ⣿⣽⠀⡀⡊⠀⠐⠨⠈⡁⠂⢈⠠⡱⡽⣷⡑⠁⠠⠑⠀⢉⢇⣤⢘⣪⢽⠀⢌⢎ ⣿⢾⠀⢌⠌⠀⡁⠢⠂⠐⡀⠀⢀⢳⢽⣽⡺⣨⢄⣑⢉⢃⢭⡲⣕⡭⣹⠠⢐⢗ ⣿⡗⠀⠢⠡⡱⡸⣔⢵⢱⢸⠈⠀⡪⣳⣳⢹⢜⡵⣱⢱⡱⣳⡹⣵⣻⢔⢅⢬⡷ ⣷⡇⡂⠡⡑⢕⢕⠕⡑⠡⢂⢊⢐⢕⡝⡮⡧⡳⣝⢴⡐⣁⠃⡫⡒⣕⢏⡮⣷⡟ ⣷⣻⣅⠑⢌⠢⠁⢐⠠⠑⡐⠐⠌⡪⠮⡫⠪⡪⡪⣺⢸⠰⠡⠠⠐⢱⠨⡪⡪⡰ ⣯⢷⣟⣇⡂⡂⡌⡀⠀⠁⡂⠅⠂⠀⡑⡄⢇⠇⢝⡨⡠⡁⢐⠠⢀⢪⡐⡜⡪⡊ ⣿⢽⡾⢹⡄⠕⡅⢇⠂⠑⣴⡬⣬...
- `105600_54613 review=11 sentence=sentence_1`: ⢀⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀⣀⣀⣤⣤⣤⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆ ⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠁⠸⣼⡿ ⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉⠀⠀⠀⠀⠀ ⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ ⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀ ⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠻⠿...
- `107410_23399 review=4 sentence=sentence_1`: Ä̸̢̢̢̛̛̛̛̝͓̺̱̻̪̗͓̜̪͙̗̱̼̲́̃̌͊͆̾̆͑̅́̓͌̄͆̄̃̆̊͗͒̃̂̈́̒͆̆̎̈́͑̉͛̃̏̇́́̓̈́͂̍̍̍͊̈́͗̔̊̋̀̐̽͒̀͂̌͐̈́̐̏̑̒̍̔̇̈̈̊͛̈́̉̓̌̅̔̍̓͌̍͛̋̀̐͑̀̽̈́̎̆͐͊́̑̒͋͐̄̍̀̇̀͊̆̊̿̐́̇̋̉͐͑̂̎̔́̍̃̂͗̏̏͋̾̏̌̉̿͒͌̾̉̀̄̌̓͋̓̔̅̈͑̔̓̇͌͋̒̀̿̑̍͋̉́̌͒̎͊̍̿̅̽͗́͋̓͛͒͆̿͂̅̃͊̔͊̃͋̈̎͂̑̈́̽̾͂̓̿̋͗̾͌̌̓̃͂̄̾̓̓́̈́̀͊̾̂̽͂̏̑̈́͒̈̒͆̉̇̈́͛͘̕̚͘͘͘̕̚̚͘͘̚͘͘̚͘͜͠͝͝͠͠͝͠͝͠w̵̧̤̼̤̳͕̫̦͕͉̖̓̓̓́̅͋̇̐̈́͂̂̂͂̏̿̾̎̐͋̈͌̏̂̈́̓͛̿́̎͒́̂̂̾͂̉̐̐́̎̈͘̚͝͝͝͝ơ̵̢̡̢̨̧̧͚̠͙̦͎͕͚̺̤̬̳̻̦̻̝͔̭͔̗̝͕̮̘͉͓̻̻̖̻̰̜̼̙͕̩̤͕̥̰̱̹̭̝̲͔̻̽͊̐́̾͂͒̆̎͐̈́͐̍̇̋̋́͋̔̇̀̈́̌̂̇͋̄̊͆̔̅̓̅̿̓̄̐͋̇̒̂́̽̄̍͂̚̕̚̚̚̕͘͜͜͜͜͠͠͝͝ͅ...
- `1150950_2354 review=3 sentence=sentence_8`: อิริยาบถน้องเดิน ก้นน้อง หางน้อง คางน้อง ลูบหัวน้อง น้องมาเกาะหัว น้องให้อุ้ม น้องงงงงงงงงงงง!!!-
- `1180320_61045 review=9 sentence=sentence_5`: ​....................../´¯/)....................,/¯../.................../..../............./´¯/'...'/´¯¯`·¸........../'/.../..../......./¨¯\........('(...´...´.... ¯~/'...').........\.................'...../..........''...\.......... _.·´............\..............(..............\.............\.
- `1180660_13083 review=4 sentence=sentence_2`: ⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀ ⣀⣀⣤⣤⣤⣀⡀⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠸⣼⡿⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇⠀⠀ ⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
- `1182620_11227 review=3 sentence=sentence_1`: . . . . . . . . . . . . . . . . . . . ,,_ . . . . . . . . ,-'`-, . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .\,, '``-.__.,.,./ .,., .\ . . . . . . . . . .. . . . . . . . . . . . . . . . . . . . .\|,.-;`;` . . ,.,., .,.,.,`-, . . . . . . . . . ...,,,.,.,._ . . . . . . . . . . . . . . / :o;. . .;o; .['. . . . `'-.,. . . . . . . . ['. . . . . . '`'`'`'`'*-----,.,.,.,._\ . . . -;- . . . . '`-,._ . . . `'-., . . . '``'*----,,,,,.,.,.,...
- `1190460_8752 review=3 sentence=sentence_1`: ⡴⠑⡄⠀⠀⠀⠀⠀⠀⠀ ⣀⣀⣤⣤⣤⣀⡀ ⠸⡇⠀⠿⡀⠀⠀⠀⣀⡴⢿⣿⣿⣿⣿⣿⣿⣿⣷⣦⡀ ⠀⠀⠀⠀⠑⢄⣠⠾⠁⣀⣄⡈⠙⣿⣿⣿⣿⣿⣿⣿⣿⣆ ⠀⠀⠀⠀⢀⡀⠁⠀⠀⠈⠙⠛⠂⠈⣿⣿⣿⣿⣿⠿⡿⢿⣆ ⠀⠀⠀⢀⡾⣁⣀⠀⠴⠂⠙⣗⡀⠀⢻⣿⣿⠭⢤⣴⣦⣤⣹⠀⠀⠀⢀⢴⣶⣆ ⠀⠀⢀⣾⣿⣿⣿⣷⣮⣽⣾⣿⣥⣴⣿⣿⡿⢂⠔⢚⡿⢿⣿⣦⣴⣾⠸⣼⡿ ⠀⢀⡞⠁⠙⠻⠿⠟⠉⠀⠛⢹⣿⣿⣿⣿⣿⣌⢤⣼⣿⣾⣿⡟⠉ ⠀⣾⣷⣶⠇⠀⠀⣤⣄⣀⡀⠈⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇ ⠀⠉⠈⠉⠀⠀⢦⡈⢻⣿⣿⣿⣶⣶⣶⣶⣤⣽⡹⣿⣿⣿⣿⡇ ⠀⠀⠀⠀⠀⠀⠀⠉⠲⣽⡻⢿⣿⣿⣿⣿⣿⣿⣷⣜⣿⣿⣿⡇ ⠀⠀ ⠀⠀⠀⠀⠀⢸⣿⣿⣷⣶⣮⣭⣽⣿⣿⣿⣿⣿⣿⣿⠇ ⠀⠀⠀⠀⠀⠀⣀⣀⣈⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠇ ⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠃.
- `1273710_8900 review=8 sentence=sentence_1`: ⣿⣿⣿⣿⡿⠋⠁⠀⠀⠀⠹⣿⣿⣿⣿⣿⣿⣿⣿⠏⠀⠀⠀⠈⠙⢿⣿⣿⣿⣿ ⣿⣿⣿⡟⠀⠀⠀⠀⠠⠀⠀⣿⣿⣿⣿⣿⣿⣿⣿⠀⠀⠄⠀⠀⠀⠀⢻⣿⣿⣿ ⣿⣿⡟⠀⠀⠀⠀⠀⠀⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣿ ⣿⣿⡇⠀⠀⠀⠀⠀⠀⢀⣮⣿⣿⣿⣿⣿⣿⣿⣿⣵⡀⠀⠀⠀⠀⠀⠀⢸⣿⣿ ⣿⣿⡀⠀⠀⠀⠀⢀⣴⣿⣯⡛⠋⠁⢻⡟⠈⠙⢛⣽⣿⣦⡀⠀⠀⠀⠀⢀⣿⣿ ⣿⣿⡇⠀⠀⠀⢾⣿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠈⠙⠻⣿⡷⠀⠀⠀⢸⣿⣿ ⠛⠛⣷⡀⠀⠀⠈⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠁⠀⠀⢀⣾⠛⠛ ⡀⠀⠈⠻⣶⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⣶⠟⠁⠀⢀ ⣷⣄⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣠⣾ ⠛⠛⠛⠶⠶⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠶⠶⠛⠛⠛ ⣆⡀⠀⠀⠀⠀⢀⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠠⡀⠀⠀⠀⠀⢀⣰ ⣿⣿⡿⠶⠚⠉⠀⠀⢀⠂⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠐⡀⠀⠀⠉⠓⠶⢿⣿⣿ ⣿⠋⠀⠀⠀⠀⣠⡴⠋⠀⠀⣠⣶⣶⣶⣶⣶⣶⣄⠀⠀⠙⢦⣄⠀⠀⠀⠀⠙⣿ ⣿⣿⣷⣾⣿⣿⣿⡇⠀⠀⢀⣿⣿⣿⣿⣿⣿⣿⣿⡀⠀⠀⢸⣿⣿⣿⣷⣾⣿⣿ ⣿⣿⣿⣿⣿⣿⣿⣷⠀⠀⣸⣿⣿...
- `1318690_12449 review=4 sentence=sentence_33`: God will then declare, "BUILD ME CbCuCbCu:Sr------:--CrSrCr:CwCwCwCw!!!"and you will say "yes, God, I will build you CbCuCbCu:Sr------:--CrSrCr:CwCwCwCw!"

### Topic 116 (58 docs)

Top words: gmod, gmod gmod, gb, gb gb, exe gmod, gmod exe, exe, g50100, g50100 g100200, gb gmod

- `1029780_17642 review=5 sentence=sentence_1`: =======大小=======✓ 百兆小游戏☐ 1-5 G☐ 5-10 G☐ 10-20 G☐ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 像素党秒天秒地！☐ 无法和现实区分✓ 赏心悦目☐ 很美丽☐ 一般☐ 不怎么样☐ 别盯着看太久☐ 画图.exe=======游戏体验=======☐ 完美！！！✓ 不错！☐ 普普通通☐ 不怎么样☐ 盯着墙看都比玩它强☐ 锻炼你的精神抗打击能力=======声音效果=======☐ 耳朵怀孕☐ 感动人心☐ 还不错✓ 一般☐ 不咋样☐ 耳朵流产☐
- `1030840_80583 review=3 sentence=sentence_1`: =======大小=======☐ 百兆小游戏☐ 1-5 G☐ 5-10 G☐ 10-20 G✓ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 像素党秒天秒地！☐ 无法和现实区分✓赏心悦目☐很美丽☐ 一般☐ 不怎么样☐ 别盯着看太久☐ 画图.exe=======游戏体验=======☐ 完美！！！☐ 不错！✓ 普普通通☐ 不怎么样☐ 盯着墙看都比玩它强☐ 锻炼你的精神抗打击能力=======声音效果=======☐ 耳朵怀孕☐感动人心☐ 还不错✓ 一般☐ 不咋样☐ 耳朵流产☐
- `1030840_80583 review=4 sentence=sentence_4`: 谁买谁二笔=======大小=======☐ 百兆小游戏☐ 1-5 G☐ 5-10 G☐ 10-20 G✓ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 像素党秒天秒地！☐ 无法和现实区分☐
- `1049890_11254 review=3 sentence=sentence_2`: 嗨~可爱的小家伙~（还不到1GB）√ 1-5 G☐ 5-10 G☐ 10-20 G☐ 20-50 G☐ 50-100 G☐ 100-200 G☐
- `1057090_26450 review=3 sentence=sentence_1`: =======大小=======☐ 百兆小游戏☐ 1-5 G☐ 5-10 G✓ 10-20 G☐ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 像素党秒天秒地！☐ 无法和现实区分✓ 赏心悦目☐ 很美丽☐ 一般☐ 不怎么样☐ 别盯着看太久☐
- `1120810_6461 review=3 sentence=sentence_1`: =======大小=======☐ 百兆小游戏☐ 1-5 G✓ 5-10 G☐ 10-20 G☐ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 像素党秒天秒地！☐ 无法和现实区分☐ 赏心悦目✓ 美丽☐ 一般☐ 不怎么样☐ 别盯着看太久☐
- `1138660_5670 review=4 sentence=sentence_1`: =======大小=======☐ 百兆小游戏✓ 1-5 G☐ 5-10 G☐ 10-20 G☐ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 像素党秒天秒地！☐ 无法和现实区分☐ 赏心悦目✓ 很美丽☐ 一般☐ 不怎么样☐ 别盯着看太久☐ 画图.exe=======游戏体验=======☐ 完美！！！✓ 不错！☐ 普普通通☐ 不怎么样☐ 盯着墙看都比玩它强☐ 锻炼你的精神抗打击能力=======声音效果=======☐ 耳朵怀孕✓ 感动人心☐ 还不错☐ 一般☐ 不咋样☐ 耳朵流产☐
- `1145360_56360 review=6 sentence=sentence_1`: =======大小=======☐ 百兆小游戏☐ 1-5 G☐ 5-10 G✓ 10-20 G☐ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 像素党秒天秒地！☐ 无法和现实区分☐ 赏心悦目✓ 很美丽☐ 一般☐ 不怎么样☐ 别盯着看太久☐
- `1145360_56360 review=8 sentence=sentence_1`: =======大小=======☐ 百兆小游戏☐ 1-5 G☐ 5-10 G✓ 10-20 G☐ 20-50 G☐ 50-100 G☐ 100-200 G☐ GMOD=======图像=======☐ 我觉得我是缸中之脑✓ Fantastic!☐ 一般☐ 画图.exe☐
- `1145360_56360 review=8 sentence=sentence_2`: 像素党秒天秒地=======游戏体验=======✓ 完美！！！☐ 不错！☐ 普普通通☐ 锻炼你的精神抗打击能力☐

### Topic 117 (58 docs)

Top words: jump, jumps, jumping, wall, fall, crystal arrow, momentum, double, slide, walls

- `1037020_2760 review=3 sentence=sentence_26`: 但是，《ScourgeBringer》的往往不能通过一般的跳跃抵达地图高处，需要依赖一个“蹬墙跑”：角色在跳到墙壁上时可以无视重力地往高处或低处匀速奔跑。
- `1037020_2760 review=3 sentence=sentence_30`: 我甚至觉得，角色在地面上主动撞墙时，应该自动蹬墙并向上跑——这样的优化在我看来非常舒服。
- `1062110_1737 review=3 sentence=sentence_33`: Once you learn which gaps are too small for a running jump it becomes easy enough to adapt to doing normal jumps for them instead.
- `1069530_1521 review=3 sentence=sentence_7`: No platformer should have such slow descent from jump.
- `1110050_1747 review=6 sentence=sentence_5`: Jumping off platforms makes them disappear and that's the core of the puzzles.
- `1130410_3627 review=3 sentence=sentence_6`: I thought I would get used to it but stuff is always just out of reach and you feel less like spider man swinging from distant buildings and more like you have one of those sticky hand toys trying to desparately fling out to something and hoping that it connects.
- `1130410_3627 review=6 sentence=sentence_4`: And once you hook somewhere from the spawn, your feet usually won't even touch the ground again for the remainder of the level.
- `1173200_2712 review=4 sentence=sentence_8`: Double jump is available from the start with a possible upgrade for a triple jump.
- `1194810_3340 review=5 sentence=sentence_13`: So any kind of jump puzzles, skill jumps, drop down, hookshot parkour or anything like that is not possible.
- `1216060_4129 review=4 sentence=sentence_24`: Some characters dont even have proper dp's and just lose the second they fall down.

### Topic 118 (57 docs)

Top words: bug bug, bug, dedsec, bgm bug, bug bugbug, game joke, fk, game strongly, preferably, fh

- `1029780_17642 review=5 sentence=sentence_8`: 如果你有点闲钱的话可以一试☐ 不推荐原价入☐ 信仰无价=======游戏整体评分=======☐ 0-10分(垃圾游戏、差评)☐ 10-30分(完完全全的不及格)☐ 30-50分(总体不是很和谐，但是买来当笑话看应该不是什么大问题)☐ 50-65分(勉强及格，算是个游戏)✓ 65-80分(算是个不错的游戏了)☐ 80-95分(无论从哪个角度来看都非常不错)☐ 95-100分(这游戏牛逼！)☐ 100-100+分(完全超出预定水准，无论从哪方面看都是精品)☐ ∞分(死忠舔狗专属评分)=======Bug=======✓ 几乎没有☐ 少量☐ bug令人烦恼☐ 育碧☐
- `1030840_80583 review=4 sentence=sentence_12`: 如果你有点闲钱的话可以一试☐ 不推荐原价入☐ 信仰无价=======游戏整体评分=======☐ 0-10分(垃圾游戏、差评)☐ 10-30分(完完全全的不及格)☐ 30-50分(总体不是很和谐，但是买来当笑话看应该不是什么大问题)☐ 50-65分(勉强及格，算是个游戏)☐ 65-80分(算是个不错的游戏了)✓ 80-95分(无论从哪个角度来看都非常不错)☐ 95-100分(这游戏牛逼！)☐ 100-100+分(完全超出预定水准，无论从哪方面看都是精品)☐ ∞分(死忠舔狗专属评分)=======Bug=======✓ 几乎没有☐ 少量☐ bug令人烦恼☐ 育碧☐
- `1037020_2760 review=3 sentence=sentence_13`: HearthStone》前设计师Ben Brode曾在一次GDC大会上分享《HearthStone》的手感奥秘：
- `1049890_11254 review=3 sentence=sentence_57`: 信仰无价---{整体评分}---☐ ？- ？分(暂时无法评定分数)☐ 0-10分(就这？就这？就这？)☐ 10-30分(不会吧？不会吧？
- `1057090_26450 review=3 sentence=sentence_8`: 如果你有点闲钱的话可以一试☐ 不推荐原价入☐ 信仰无价=======游戏整体评分=======☐ 0-10分(垃圾游戏、差评)☐ 10-30分(完完全全的不及格)☐ 30-50分(总体不是很和谐，但是买来当笑话看应该不是什么大问题)☐ 50-65分(勉强及格，算是个游戏)☐ 65-80分(算是个不错的游戏了)☐ 80-95分(无论从哪个角度来看都非常不错)☐ 95-100分(这游戏牛逼！)☐ 100-100+分(完全超出预定水准，无论从哪方面看都是精品)✓ ∞分(死忠舔狗专属评分)=======Bug=======✓ 几乎没有☐ 少量☐ bug令人烦恼☐ 育碧☐
- `1220010_6818 review=5 sentence=sentence_34`: 信仰无价=======游戏整体评分=======☐ ？- ？分(暂时无法评定分数)☐ 0-10分(这是什么JB玩意！)☐ 10-30分(完完全全的不及格)☐ 30-50分(毛病比优点多)☐ 50-65分(勉强及格，看得出来这是个游戏)☐ 65-80分(算是个可以一玩的游戏了)☑ 80-95分(有些小瑕疵，但综合来看都非常不错)☐ 95-100分(同类游戏当之无愧的第一名！)☐ 100+分(市面上不存在任何替代产品)☐ ∞分(死忠舔狗专属评分)
- `1237320_22456 review=4 sentence=sentence_29`: 这个死循环，我曾经在育碧的罐头式开放世界体验过同样感觉，一大片各种问号，你做的每一个任务都是套个皮重复的，不断去循环刷据点，刷完过后只剩下疲劳感，一点“开放世界”的探索乐趣都没有。
- `1282690_5495 review=3 sentence=sentence_2`: 在1999年的e3展上，他们用自己的展示游戏《岛屿x》里远超当时游戏水准的画质引起了游戏界的广泛关注，而后来育碧也与他们合作把这游戏演示demo变成了一个真正的游戏并用Cryengine引擎制作，这个游戏便是后来著名的射击游戏《Far Cry（远哭）》
- `1293830_263270 review=4 sentence=sentence_7`: 如果你有点闲钱的话可以一试☐ 不推荐原价入☐ 信仰无价=======游戏整体评分=======☐ 0-10分(垃圾游戏、差评)☐ 10-30分(完完全全的不及格)☐ 30-50分(总体不是很和谐，但是买来当笑话看应该不是什么大问题)☐ 50-65分(勉强及格，算是个游戏)☐ 65-80分(算是个不错的游戏了)✓ 80-95分(无论从哪个角度来看都非常不错)☐ 95-100分(这游戏牛逼！)☐ 100-100+分(完全超出预定水准，无论从哪方面看都是精品)☐ ∞分(死忠舔狗专属评分)=======Bug=======✓ 几乎没有☐ 少量☐ bug令人烦恼☐ 育碧☐
- `1330470_6244 review=5 sentence=sentence_22`: 如果你有点闲钱的话可以一试☐ 不推荐原价入☐ 信仰无价---{游戏整体评分}---☐ ？- ？分(暂时无法评定分数)☐ 0-10分(垃圾游戏、差评)☐ 10-30分(完完全全的不及格)☐ 30-50分(总体不是很和谐，但是买来当笑话看应该不是什么大问题)☐ 50-65分(勉强及格，算是个游戏)☐ 65-80分(算是个不错的游戏了)✓ 80-95分(无论从哪个角度来看都非常不错)☐ 95-100分(这游戏牛逼！)☐ 100-100+分(完全超出预定水准，无论从哪方面看都是精品)☐ ∞分(死忠舔狗专属评分)---{Bug}---☐ 几乎没有✓少量☐ 令人烦恼☐ 育碧☐

### Topic 119 (56 docs)

Top words: chinese need, need chinese, chinese, pleasekorean, pleasekorean pleasekorean, need, china, republic, scenario, war ii

- `1138660_5670 review=5 sentence=sentence_1`: Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean please!!!!!!!!!!!!!!!!Korean pleas...
- `1302240_10730 review=8 sentence=sentence_1`: 中文求求了，Chinese Please，Playing with friends,have no Chinese I feel less atmosphere .
- `1694600_2128 review=3 sentence=sentence_6`: PLEASE PLEASE PLEASE!!!
- `2096600_5421 review=3 sentence=sentence_23`: At least Prophet told us that we urgently need to contact the scientist Dr. Gould should go because he could help us.
- `2772750_6846 review=3 sentence=sentence_38`: The Republic of China in the World War II scenario has the same flag as Manchuria and has the same name as the People's Republic of China \| Fixed+ Game Crashing after 10 minutes on 1610 scenario idk why
- `402710_14890 review=4 sentence=sentence_5`: But no more chinese ..
- `508440_4150 review=3 sentence=sentence_14`: Hurry up to sinicize!!!!!
- `758870_3013 review=3 sentence=sentence_1`: We need Chinese!!!!! Please!!!!!!
- `758870_3013 review=3 sentence=sentence_2`: We need Chinese!!!!! Please!!!!!!
- `758870_3013 review=3 sentence=sentence_3`: We need Chinese!!!!! Please!!!!!!

### Topic 120 (56 docs)

Top words: visual, visual novel, novel, novels, visual novels, digimon, musicals, novel game, story, great visual

- `1000030_2098 review=3 sentence=sentence_19`: The story is not on the same level as the likes of LISA or Disco Elysium, though it is serviceable and interesting (I definitely did some Wiki deepdiving to find out about The Blue War and what happened to Japan).
- `1129190_11692 review=4 sentence=sentence_12`: I have never played a visual novel before and I was gripped to it the entire time.
- `1140290_1513 review=4 sentence=sentence_3`: This is a *visual novel* with picross, and one with no real choices at that.
- `1140290_1513 review=4 sentence=sentence_5`: I've got nothing against visual novels, but the characters, writing, and plot were just not for me.
- `1150080_7309 review=3 sentence=sentence_9`: The game has a lot of visual novel content and less when it comes to gameplay.
- `1150080_7309 review=3 sentence=sentence_23`: Crosswave content lies in its visual novel, not it's the gameplay.
- `1163550_5359 review=4 sentence=sentence_17`: The story plays itself in a nice 3D visual novel style with some dialogue choices for the player.
- `1202900_7556 review=6 sentence=sentence_5`: Story was nice, but if I want just story and arts I go for a visual novel game, here I wanted actual puzzle gameplay promised in screenshots.
- `1269640_3067 review=4 sentence=sentence_27`: Highly Recommended for fans of Night in the Wood, the 1980s, narrative adventures and visual novels
- `1274290_6350 review=10 sentence=sentence_1`: - DIFFICULTY -☑️ Its Visual Novel

### Topic 121 (55 docs)

Top words: character, notre, create character, customize, mask, customize character, body, gender, customization, changer

- `1062810_3287 review=3 sentence=sentence_13`: I also love the simplicity of the character customization being basically just a face painting simulator, allowing players to create very unique looking faces.
- `1069640_4384 review=5 sentence=sentence_26`: Charakterentwicklung: Eine Kombination aus vergebenem Namen und einem „Totem“, hier etwas experimentieren.
- `1114220_3476 review=3 sentence=sentence_4`: No universal do all character development but allows you to play a wizard and become a great wizard...
- `1167450_6168 review=3 sentence=sentence_7`: It goes for everything - customization, combat, world building, story, even characters, despite them having faces now.
- `1191580_1556 review=3 sentence=sentence_9`: Pour la création de notre partie on peut choisir entre Fille ou Garçon, mais directement dans le jeu, via notre profile dans les options on peut changer notre skin et changer de genre, et même de changer notre apparence si celles de départ nous convient pas, et c'est gratuit, directement mis à notre disposition, de même qu'on peut ajouter si on veut un Nexomon qui peut nous suivre (même si on le possède pas) pour le coté esthétique mais le cho...
- `1227530_7844 review=5 sentence=sentence_6`: 제작자들 플레이 안해봤나?다섯번째 고유의 특성이 게임이 자랑질했던 것 중 하나인데캐릭터마다 스킬이 조금 달라서 개성넘치는 커스터마이징을 할 수 있다고 했음시발
- `1238040_5837 review=4 sentence=sentence_2`: Можно настроить лицо героя.
- `1264250_2449 review=3 sentence=sentence_11`: (+)-Oyuncu Karakteri; Oyunda kendinize bir profil oluşturabiliyorsunuz adınız ve cinsiyetinizi ekliyebiliyorsunuz.
- `1343240_7595 review=4 sentence=sentence_8`: No hay personalización de personaje, tanto de apariencia como de armas (no es demasiado importante ya que cuentan la historia de Corvus y pues tiene sentido jugar siendo EL intrínsecamente )-
- `1490890_18517 review=3 sentence=sentence_43`: Even a more customization option can really make your character pop out more.

### Topic 122 (54 docs)

Top words: significant brain, brain usage, brain, usage, significant, usage significant, uso significativo, significativo, uso, significativo crebro

- `1029550_11568 review=3 sentence=sentence_16`: Significant brain usage☐
- `1082430_15462 review=6 sentence=sentence_24`: Significant brain usage☐
- `1106840_11729 review=7 sentence=sentence_14`: Significant brain usage☐
- `1172470_50120 review=3 sentence=sentence_15`: Use your brain, which is not yet as big as a melon seed, to think carefully.
- `1201240_11231 review=6 sentence=sentence_14`: Significant brain usage☐
- `1216320_6022 review=7 sentence=sentence_13`: Significant brain usage☐
- `1229490_53960 review=3 sentence=sentence_1`: ULTRAKILL has literally rewired the physical, biological wiring of my ADHD-addled brain and body to incalculable speeds at which even polypropylene-film capacitors cannot compare.
- `1229490_53960 review=3 sentence=sentence_18`: It awakens my brain up to function for the rest of the day to a level that coffee or orange juice simply cannot compare.
- `1260520_3670 review=5 sentence=sentence_23`: So: do you want a brain workout, or would you like to just be taken on a magic box ride through a theme park of guided "aha" moments?
- `1273710_8900 review=4 sentence=sentence_14`: Significant brain usage☐

### Topic 123 (53 docs)

Top words: fun game, fun, really fun, game fun, lot fun, game lot, damn fun, game damn, genuinely fun, extremely fun

- `1003590_7381 review=6 sentence=sentence_1`: テトリスはプレイしているとぐっと集中して何も他に考えられなくなるような、ある種の中毒性を持ったゲームです。
- `1040200_11371 review=3 sentence=sentence_7`: 그러니 게임 플레이를 하시면서 이러한 말투를 분석 하시면 더욱 재미있는 게임 플레이가 될 겁니다!
- `1089090_19106 review=7 sentence=sentence_19`: ─ Gameplay excelente e divertida;
- `1128920_10841 review=7 sentence=sentence_11`: It's fun to play, and wow did Rockfish games get a lot right.
- `1129310_2070 review=4 sentence=sentence_45`: This game, on the other hand, is quite happy to walk the walk.
- `1135230_5454 review=6 sentence=sentence_1`: This game is extremely fun!
- `1177980_7488 review=7 sentence=sentence_3`: I'll say this right out the gate: this game is about as wholesome an experience as you can find in gaming and is a game that can be enjoyed equally by children and adults, making it an obvious title for people also wanting to have some gaming time with their kids, amongst other things.
- `1244800_2991 review=4 sentence=sentence_1`: This game is surprisingly fun.
- `1271700_7057 review=3 sentence=sentence_24`: This game is a LOT of fun.
- `1285670_10427 review=5 sentence=sentence_5`: The game is really fun.

### Topic 124 (53 docs)

Top words: buy game, worth, game worth, buy, pena, vale, thinking buying, comprar, definitely worth, game definitely

- `1041720_6700 review=6 sentence=sentence_13`: Uma boa semana a todos vocês e espero ter ajudado caso possuissem alguma dúvida se valia ou não comprar o game.
- `1069640_4384 review=3 sentence=sentence_1`: Is the game worth it?
- `1082430_15462 review=4 sentence=sentence_3`: 3- Buy the game, anyways.
- `1113000_73340 review=4 sentence=sentence_3`: If this helps sway even a single person to get this game, then it’ll be worth it.
- `1119730_16960 review=6 sentence=sentence_9`: 이 게임을 구매하는 것에 대해 다시 한번 생각해 보십시오.
- `1140270_5797 review=4 sentence=sentence_20`: That's worth buying this game for.3.
- `1145960_5728 review=3 sentence=sentence_14`: Does that mean you shouldn't buy and play it?
- `1166860_3871 review=3 sentence=sentence_12`: If we got that content, it might make this game worth it.
- `1201540_3395 review=4 sentence=sentence_16`: If you're already thinking about buying this game, just buy it.
- `1286830_1160 review=4 sentence=sentence_17`: It's a thing to consider, but if you are reading this review, you are probably wondering if it's worth picking up the game.

### Topic 125 (52 docs)

Top words: eargasm good, bad bad, eargasm, good bad, bad, audio eargasm, dont audio, good good, audio, just dont

- `1029550_11568 review=3 sentence=sentence_8`: I'm now deaf---{ 🧑Audience🧑 }---☑ Kids☑ Teens☑ Adults☑
- `1082430_15462 review=6 sentence=sentence_17`: Earrape---{Audience}---☐ Kids☐ Teens☐ Adults☑ Human☐
- `1106840_11729 review=7 sentence=sentence_8`: Just don't---{ Audio }---☐ Eargasm☐ Very good☐ Good☑ Not too bad☐ Bad☐
- `1201240_11231 review=6 sentence=sentence_8`: Just don't---{ Audio }---☐ Eargasm☐ Very good☑ Good☐ Not too bad☐ Bad☐
- `1201270_5269 review=4 sentence=sentence_6`: Sampe Mau Nangis🔵 Suara 🔵🔲 Earrape sampe budek / Gak ada suara🔲 Jelek🔲 Rata Rata✅ Bagus🔲
- `1216320_6022 review=7 sentence=sentence_8`: Just don't---{ Audio }---☐ Eargasm☐ Very good☐ Good☑ Not too bad☐ Bad☐
- `1273710_8900 review=4 sentence=sentence_8`: Just don't---{ Audio }---☐ Eargasm☐ Very good☐ Good☑ Not too bad☐ Bad☐
- `1274290_6350 review=6 sentence=sentence_12`: I'm now deaf---{Audience}---☐ Kids☐ Teens✅ Adults☐
- `1280770_16276 review=7 sentence=sentence_8`: Just don't---{ Audio }---☐ Eargasm☐ Very good☐ Good☑ Not too bad☐ Bad☐
- `1282150_3166 review=10 sentence=sentence_16`: Horrible---{Audiencia}---☑ Para bebes☐ Adolecentes con granos☐

### Topic 126 (52 docs)

Top words: rich ladys, ladys slave, slave role, leaderboardsranks, care leaderboardsranks, ladys, role play, slave, rich, role

- `1082430_15462 review=6 sentence=sentence_27`: Only if u care about leaderboards/ranks☐
- `1097430_6948 review=3 sentence=sentence_23`: 千金的奴隶家家酒：Rich Lady's Slave Role Play
- `1106840_11729 review=7 sentence=sentence_16`: Only if u care about leaderboards/ranks☐
- `1144400_27470 review=3 sentence=sentence_23`: 千金的奴隶家家酒：Rich Lady's Slave Role Play
- `1146630_6863 review=5 sentence=sentence_23`: 千金的奴隶家家酒：Rich Lady's Slave Role Play
- `1201240_11231 review=6 sentence=sentence_16`: Only if u care about leaderboards/ranks☐
- `1213740_18053 review=3 sentence=sentence_21`: 千金的奴隶家家酒：Rich Lady's Slave Role Play精灵之妊 -用怀孕征服所有傲慢的精灵与经纪人恋爱是绝对禁止2圣石少女篇：OVER‧DeviL恶魔调酒师九点开张快捷情趣酒店：Quickie:A Love Hotel Story
- `1273710_8900 review=4 sentence=sentence_18`: Only if u care about leaderboards/ranks☐
- `1274290_6350 review=6 sentence=sentence_23`: Only if u care about leaderboards/ranks☐
- `1280770_16276 review=7 sentence=sentence_15`: Only if u care about leaderboards/ranks☐
