from flask_appbuilder import Model
from sqlalchemy import Column, Integer, String, ForeignKey, UniqueConstraint
from sqlalchemy.orm import relationship
from myapp.models.base import MyappModelBase


class UserCustomPermission(Model, MyappModelBase):
    __tablename__ = "user_custom_permission"
    id = Column(Integer, primary_key=True, comment="主键")
    user_id = Column(Integer, ForeignKey("ab_user.id"), nullable=False, comment="用户ID")
    view_menu_name = Column(String(250), nullable=False, comment="接口名")

    user = relationship("MyUser", backref="custom_permissions", foreign_keys=[user_id])

    __table_args__ = (
        UniqueConstraint("user_id", "view_menu_name", name="uq_user_view"),
    )

    def __repr__(self):
        return f"{self.view_menu_name}:{self.permission_name}"
