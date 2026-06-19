from django.db import models


class SpendEvent(models.Model):
    """A single spend event within a paycheck period."""

    paycheck = models.ForeignKey(
        "paychecks.Paycheck",
        on_delete=models.CASCADE,
        related_name="spend_events",
    )
    amount = models.FloatField()
    category = models.CharField(max_length=50)
    description = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        db_table = "spend_events"


class BufferBalance(models.Model):
    """Running carry-forward buffer per user."""

    user = models.OneToOneField(
        "users.User",
        on_delete=models.CASCADE,
        related_name="buffer",
    )
    balance = models.FloatField(default=0.0)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        db_table = "buffer_balances"
