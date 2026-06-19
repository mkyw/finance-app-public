from django.db import models


class HouseholdProfileRecord(models.Model):
    """Persisted onboarding profile; mirrors shared.types.HouseholdProfile."""

    user = models.OneToOneField(
        "users.User",
        on_delete=models.CASCADE,
        related_name="profile",
    )
    age = models.IntegerField()
    gross_income = models.FloatField()
    puma_code = models.CharField(max_length=20)
    tenure = models.CharField(
        max_length=4,
        choices=[("OWN", "Own"), ("RENT", "Rent")],
    )
    housing_cost = models.FloatField()
    household_size = models.IntegerField()
    filing_status = models.CharField(max_length=30, default="single")
    financial_zone = models.CharField(max_length=20, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "household_profiles"
