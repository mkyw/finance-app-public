from django.db import models


class BenefitEligibilityRecord(models.Model):
    """Persisted output of models.benefits.eligibility.screen."""

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="benefit_eligibility",
    )
    program_name = models.CharField(max_length=100)
    estimated_monthly_min = models.FloatField()
    estimated_monthly_max = models.FloatField()
    confidence = models.CharField(max_length=20)
    enrollment_url = models.URLField()
    screened_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "benefit_eligibility"
