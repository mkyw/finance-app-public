from django.db import models


class Paycheck(models.Model):
    """One paycheck period per row; parent of SpendEvent rows."""

    user = models.ForeignKey(
        "users.User",
        on_delete=models.CASCADE,
        related_name="paychecks",
    )
    gross_amount = models.FloatField()
    pay_date = models.DateField()
    next_pay_date = models.DateField()
    discretionary_available = models.FloatField()
    current_spend = models.FloatField(default=0.0)
    pace_projection = models.FloatField(null=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "paychecks"
