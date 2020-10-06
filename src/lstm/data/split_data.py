
        lang_np = np.array(languages)
        print(lang_np.shape)
        #lines = np.array(lines)
        langs = list(set(languages))
        langs.sort()
        print(len(langs))
        val_set_idx = []
        train_set_idx = []
        for i, language in enumerate(langs):
            print(language)
            language_idx = np.where(lang_np==language)[0]
            n_points_langs = language_idx.shape[0]
            print("languages: ", n_points_langs)
            random_set = np.random.permutation(n_points_langs)
            val = language_idx[random_set[:int(n_points_langs*0.2)]]
            train = language_idx[random_set[int(n_points_langs*0.2):]]
            val_set_idx.append(val)
            train_set_idx.append(train)
            print(i)


        #raise ValueError()
        with open("x_val_sub_clean.txt", 'w') as f:
            for val_idx in val_set_idx:
                for idx in val_idx:
                    para = lines[idx]
                    f.write(para)

        with open("y_val_sub_clean.txt", 'w') as f:
            for val_idx in val_set_idx:
                y_subset = lang_np[val_idx]
                for lang in y_subset:
                        f.write(lang)


        with open("x_train_sub_clean.txt", 'w') as f:
            for train_idx in train_set_idx:
                for idx in train_idx:
                    para = lines[idx]
                    f.write(para)

        with open("y_train_sub_clean.txt", 'w') as f:
            for train_idx in train_set_idx:
                y_subset = lang_np[train_idx]
                for lang in y_subset:
                        f.write(lang)
