    pairs = [("apple", ["pecan", "pear","apples","peach"]),
             ("gay", ["debonair", "carefree", "bisexual", "activists"]),
             ("mail", ["cable", "postal", "e", "stationery"] ),
             ("canadian",["port", "railhead", "federal","national"]),
             ("cell", ["prison", "dungeon", "pager", "handset"])
             ]
    for word,candidates in pairs:
        print(word)
        for year in [1920,1960,2000]:
            baisc = model.get_temporal_embedding([word],[year])[0]
            candidates_embedding = model.get_temporal_embedding(candidates,[year])
            sims = [ np.inner(baisc,c)  for c in candidates_embedding]
            print(sims)
