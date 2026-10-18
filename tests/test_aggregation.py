import pytest
from src.aggregation import aggregate_scores
from src.segmentation import category
def test_categories():assert [category(0,0),category(1,0),category(0,1),category(1,1)]==['Normal','Crackles','Wheezes','Both']
def test_aggregation():
 r=aggregate_scores([[.8,.2],[.4,.7]]);assert r['category']=='Both' and r['crackles_score']==.8
def test_invalid_scores():
 with pytest.raises(ValueError):aggregate_scores([[2,0]])
