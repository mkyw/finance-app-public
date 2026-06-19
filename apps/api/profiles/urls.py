from django.urls import path

from .views import CityResolveView, ProfileAnalysisView

urlpatterns = [
    path("analyze/", ProfileAnalysisView.as_view(), name="profile-analyze"),
    path("resolve-city/", CityResolveView.as_view(), name="resolve-city"),
]
