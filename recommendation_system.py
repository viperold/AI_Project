"""
recommendation_system.py

Ejemplo inicial de sistema de recomendación híbrido (content-based + colaborativo)
Incluye: generación de datos sintéticos, funciones de recomendación y evaluación simple.

Ejecutar:
	python recommendation_system.py

Requiere: pandas, numpy, scikit-learn
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


@dataclass
class Rating:
	user_id: int
	item_id: int
	rating: float


class SimpleRecommender:
	"""Recomendador híbrido simple.

	- Content-based: usa vectores de características de items y similitud coseno.
	- Collaborative (neighbourhood): usa la matriz usuario-item y similitud entre usuarios.
	"""

	def __init__(self, ratings: pd.DataFrame, items: pd.DataFrame):
		self.ratings = ratings.copy()
		self.items = items.copy()
		self.user_item_matrix = self._build_user_item_matrix()

	def _build_user_item_matrix(self) -> pd.DataFrame:
		matrix = self.ratings.pivot_table(index="user_id", columns="item_id", values="rating")
		return matrix

	def recommend_content_based(self, user_id: int, top_k: int = 5) -> List[Tuple[int, float]]:
		"""Recomienda ítems similares a los ya valorados por el usuario.

		Devuelve una lista de tuplas (item_id, score)
		"""
		if user_id not in self.user_item_matrix.index:
			return []

		# obtener items valorados por el usuario
		user_ratings = self.user_item_matrix.loc[user_id].dropna()
		rated_item_ids = user_ratings.index.tolist()

		# construir matriz de características de items
		item_features = self.items.set_index("item_id").drop(columns=["item_id"], errors="ignore")

		# similaridad entre todos los items
		sim = cosine_similarity(item_features)
		sim_df = pd.DataFrame(sim, index=item_features.index, columns=item_features.index)

		# puntuar items no valorados por promedio ponderado de similitudes
		scores: Dict[int, float] = {}
		for candidate in item_features.index.difference(rated_item_ids):
			# similitud con cada ítem valorado por el usuario
			s = 0.0
			weight = 0.0
			for iid, r in user_ratings.items():
				s += sim_df.at[candidate, iid] * r
				weight += abs(sim_df.at[candidate, iid])
			scores[candidate] = (s / weight) if weight != 0 else 0.0

		ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
		return ranked

	def recommend_user_based(self, user_id: int, top_k: int = 5, n_neighbors: int = 5) -> List[Tuple[int, float]]:
		"""Recomendación colaborativa basada en similitud de usuarios.

		Calcula similitud coseno entre usuarios y promedia las valoraciones de vecinos.
		"""
		matrix = self.user_item_matrix.fillna(0)
		if user_id not in matrix.index:
			return []

		user_vec = matrix.loc[user_id].values.reshape(1, -1)
		sims = cosine_similarity(user_vec, matrix.values)[0]
		sim_series = pd.Series(sims, index=matrix.index)
		sim_series = sim_series.drop(user_id)

		neighbors = sim_series.nlargest(n_neighbors)

		# puntuación: media ponderada de valoraciones de vecinos
		scores: Dict[int, float] = {}
		for item in matrix.columns:
			if not np.isnan(self.user_item_matrix.at[user_id, item]) if item in self.user_item_matrix.columns else True:
				# saltar items ya valorados
				if item in self.user_item_matrix.columns and not pd.isna(self.user_item_matrix.at[user_id, item]):
					continue

			num = 0.0
			den = 0.0
			for neighbor_id, sim in neighbors.items():
				rating = matrix.at[neighbor_id, item]
				if rating != 0:
					num += sim * rating
					den += abs(sim)
			if den != 0:
				scores[item] = num / den

		ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
		return ranked


def generate_synthetic_data(num_users: int = 50, num_items: int = 100, sparsity: float = 0.05, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
	random.seed(random_state)
	np.random.seed(random_state)

	# generar características de items (por ejemplo 5 características float)
	item_ids = list(range(1, num_items + 1))
	features = np.random.rand(num_items, 5)
	items = pd.DataFrame(features, columns=[f"f{i}" for i in range(1, 6)])
	items["item_id"] = item_ids
	items = items[["item_id"] + [c for c in items.columns if c != "item_id"]]

	# generar ratings dispersos
	rows: List[Rating] = []
	for user in range(1, num_users + 1):
		for item in item_ids:
			if random.random() < sparsity:
				rating = round(random.uniform(1, 5), 1)
				rows.append(Rating(user_id=user, item_id=item, rating=rating))

	ratings = pd.DataFrame([r.__dict__ for r in rows])
	return ratings, items


def precision_at_k(recommended: List[int], relevant: List[int], k: int) -> float:
	if not recommended:
		return 0.0
	recommended_k = recommended[:k]
	hits = len(set(recommended_k) & set(relevant))
	return hits / k


def demo():
	ratings, items = generate_synthetic_data(num_users=30, num_items=80, sparsity=0.08)
	model = SimpleRecommender(ratings=ratings, items=items)

	# elegir un usuario con al menos una valoración
	users_with_ratings = ratings.user_id.unique().tolist()
	test_user = users_with_ratings[0]

	print(f"Usuario de demo: {test_user}")
	print("Valoraciones del usuario:")
	print(ratings[ratings.user_id == test_user].head())

	cb = model.recommend_content_based(test_user, top_k=5)
	ub = model.recommend_user_based(test_user, top_k=5)

	print("\nRecomendaciones Content-based:")
	for item_id, score in cb:
		print(f"  Item {item_id} — score {score:.3f}")

	print("\nRecomendaciones User-based:")
	for item_id, score in ub:
		print(f"  Item {item_id} — score {score:.3f}")


if __name__ == "__main__":
	demo()
