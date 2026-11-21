"""Issue reports: any signed-in person can send one."""
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.api.dependencies import get_session, signed_in_user
from app.core import audit
from app.db.models import IssueReport, User
from app.schemas.issues import IssueReportCreate, IssueReportOut

router = APIRouter(tags=["issues"])


@router.post("/issue-reports", response_model=IssueReportOut, status_code=status.HTTP_201_CREATED)
def report_issue(
    body: IssueReportCreate,
    user: User = Depends(signed_in_user),
    session: Session = Depends(get_session),
):
    report = IssueReport(user_id=user.id, kind=body.kind, details=body.details.strip())
    session.add(report)
    session.flush()
    # Only the length is logged: the details are the person's own words.
    audit.event(
        "issue_report", actor=user.email, report=report.id, kind=body.kind,
        details_length=len(report.details),
    )
    return IssueReportOut(id=str(report.id), created_at=report.created_at.isoformat())
