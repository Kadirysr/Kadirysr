
#############################################
# Adım 1: Veri Setinin Hazırlanması
#############################################
import pandas as pd

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
#movie ve rating için left join gerçekleştiriliyor.
df = movie.merge(rating, how="left", on="movieId")

#eşsiz film sayısı
df["title"].nunique()
#hangi filme kaç tane puan verilmiş.
df["title"].value_counts().head()
#filmlerin kaç tane puan verildiği.
rating_counts = pd.DataFrame(df["title"].value_counts())
#1000 den az puanlaması olan movies.
rare_movies = rating_counts[rating_counts["title"] <= 1000].index
#1000 den az olanları çıkarıyoruz.
common_movies = df[~df["title"].isin(rare_movies)]
common_movies.head()
#3159 movie sayısı.
common_movies["title"].nunique()

user_movie_df = common_movies.pivot_table(index=["userId"], columns=["title"], values="rating")

user_movie_df.head(2)
#pivot yerine aşağıdaki de kullanılabilir.
#user_movie_df2 =common_movies.groupby(['userId', 'title'])['rating'].max().unstack()

#random_user = int(pd.Series(user_movie_df.index).sample(1, random_state=45).values)

#Görev 2:Öneri yapılacak kullanıcının izlediği filmleri belirleyiniz.

random_user = 108170
random_user_df = user_movie_df[user_movie_df.index == random_user]


movies_watched = random_user_df.columns[random_user_df.notna().any()].tolist()

user_movie_df.loc[user_movie_df.index == random_user, user_movie_df.columns == "Schindler's List (1993)"]

len(movies_watched)

#Görev 3: Aynı filmleri izleyen diğer kullanıcıların verisine ve Id'lerine erişiniz.

#seçilen userin izlediği filmleri user_movie_df ten seçiyoruz
movies_watched_df = user_movie_df[movies_watched]
movies_watched_df.head()

movies_watched_df.shape
#her bir kullanıcı için filmlerin kaçını izlediği hesaplanıyor.
user_movie_count = movies_watched_df.T.notnull().sum()
user_movie_count = user_movie_count.reset_index()
user_movie_count.columns = ["userId", "movie_count"]
user_movie_count.head()

#Görev 4: Öneri yapılacak kullanıcı ile en benzer kullanıcıları belirleyiniz.
#%60 izleyenleri seçiyoruz.
perc = len(movies_watched) * 60 /100
users_same_movies = user_movie_count[user_movie_count["movie_count"] > perc]["userId"]
#users_same_movies = user_movie_count[user_movie_count["movie_count"] > 20]["userId"]

users_same_movies.head()
users_same_movies.count()
users_same_movies.index

final_df = movies_watched_df[movies_watched_df.index.isin(users_same_movies)]
final_df.head
final_df.T
#corr ile filmlerin benzerliği bulunur. correlation sütunlar arasında olur.
final_df.corr()
#transpose yapılır. filmler satırlara, idler sütunlara alınmış olur.
final_df.T

corr_df = final_df.T.corr().unstack().sort_values()
corr_df.head()
corr_df = pd.DataFrame(corr_df, columns=["corr"])
corr_df.head()
corr_df.index.names = ['user_id_1', 'user_id_2']
corr_df =corr_df.reset_index()
corr_df.head()

top_users = corr_df[(corr_df["user_id_1"] == random_user) & (corr_df["corr"] >= 0.65)][
    ["user_id_2", "corr"]].reset_index(drop=True)

top_users = top_users.sort_values(by='corr', ascending=False)

top_users.rename(columns={"user_id_2": "userId"}, inplace=True)

top_users

rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')
top_users_ratings = top_users.merge(rating[["userId", "movieId", "rating"]], how='inner')

top_users_ratings.head(50)

top_users_ratings = top_users_ratings[top_users_ratings["userId"] != random_user]
top_users_ratings.head()

#Görev 5:Weighted Average Recommendation Score'u hesaplayınız ve ilk 5 filmi tutunuz.
top_users_ratings['weighted_rating'] = top_users_ratings['corr'] * top_users_ratings['rating']


top_users_ratings.head()


top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})

recommendation_df = top_users_ratings.groupby('movieId').agg({"weighted_rating": "mean"})
recommendation_df = recommendation_df.reset_index()

recommendation_df.head()

recommendation_df[["movieId"]].nunique()

recommendation_df[recommendation_df["weighted_rating"] > 3]

movies_to_be_recommend = recommendation_df[recommendation_df["weighted_rating"] > 3].sort_values("weighted_rating", ascending=False)
movies_to_be_recommend.head()
movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
movies_to_be_recommend.merge(movie[["movieId", "title"]])["title"].head(5)

# Görev 6: Kullanıcının izlediği filmlerden en son en yüksek puan verdiği filmin adına göre item-based öneri yapınız.

user = 108170

movie = pd.read_csv('datasets/movie_lens_dataset/movie.csv')
rating = pd.read_csv('datasets/movie_lens_dataset/rating.csv')

movie.head()
# Öneri yapılacak kullanıcının 5 puan verdiği filmlerden puanı en güncel olan filmin id'sinin alınması:
movie_id = rating[(rating["userId"] == user) & (rating["rating"] == 5.0)].sort_values(by="timestamp", ascending=False)["movieId"][0:1].values[0]


def item_based_recommender(movie_name, user_movie_df):
    movie = user_movie_df[movie_name]
    return user_movie_df.corrwith(movie).sort_values(ascending=False).head(10)

movies_from_item_based = item_based_recommender(movie[movie["movieId"] == movie_id]["title"].values[0], user_movie_df)
# 1'den 6'ya kadar. 0'da filmin kendisi var. Onu dışarda bıraktık.
movies_from_item_based[1:6].index

item_based_recommender("¡Three Amigos! (1986)", user_movie_df)

