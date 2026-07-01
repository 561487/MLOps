import re

from myapp import db, security_manager
from myapp.models.model_storage import Storage
from myapp.models.model_team import Project_User


def split_volume_mount(value):
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(split_volume_mount(item))
        return values
    value = value or ''
    return [item.strip() for item in re.split(',|;|\n|\t', value) if item and item.strip()]


def parse_volume_mount(expr):
    expr = (expr or '').strip()
    match = re.match(r'^(.+?)\(([^()]+)\):(.+)$', expr)
    if not match:
        return None
    name = match.group(1).strip()
    volume_type = match.group(2).strip()
    mount_path = match.group(3).strip()
    if not name or not volume_type or not mount_path:
        return None
    return {
        "name": name,
        "volume_type": volume_type,
        "mount_path": mount_path,
        "mount_expr": expr,
    }


def user_can_access_project(user, project):
    if not user or not project:
        return False
    if user.is_admin():
        return True
    user_id = user.get_id()
    return db.session.query(Project_User.id).filter(
        Project_User.project_id == project.id,
        Project_User.user_id == user_id
    ).first() is not None


def _joined_project_ids(user):
    if not user:
        return []
    if user.is_admin():
        return []
    try:
        return security_manager.get_join_projects_id(db.session)
    except Exception:
        return [
            item[0] for item in db.session.query(Project_User.project_id).filter(
                Project_User.user_id == user.get_id()
            ).all()
        ]


def _storage_is_ready(storage):
    if storage.status in ['synced', 'shared_verified']:
        return True
    if storage.status in ['draft', 'error', 'deleted', 'backend_mismatch']:
        return False
    return bool(storage.pvc_name)


def _append_unique(items, item, seen_pvc, seen_mount):
    parsed = parse_volume_mount(item.get('mount_expr', ''))
    if not parsed:
        return
    pvc_key = (parsed['name'], parsed['volume_type'])
    mount_key = parsed['mount_path'].rstrip('/') or '/'
    if pvc_key in seen_pvc or mount_key in seen_mount:
        return
    seen_pvc.add(pvc_key)
    seen_mount.add(mount_key)
    items.append(item)


def _storage_mount_path(storage):
    mount_path = (storage.mount_path or '').strip().rstrip('/')
    if not mount_path or mount_path == '/mnt/storage':
        return '/mnt/storage/{}'.format(storage.name)
    return mount_path


def available_volume_items(user, project=None, namespace=None, current_volume_mount='', include_system=True):
    checked_exprs = set(split_volume_mount(current_volume_mount))
    items = []
    seen_pvc = set()
    seen_mount = set()

    if project and include_system:
        for expr in split_volume_mount(project.volume_mount):
            parsed = parse_volume_mount(expr)
            if not parsed:
                continue
            _append_unique(items, {
                "name": parsed['name'],
                "label": parsed['name'],
                "source": "system",
                "volume_type": parsed['volume_type'],
                "mount_path": parsed['mount_path'],
                "mount_expr": parsed['mount_expr'],
                "checked": parsed['mount_expr'] in checked_exprs or not checked_exprs,
            }, seen_pvc, seen_mount)

    query = db.session.query(Storage)
    if project:
        query = query.filter(Storage.project_id == project.id)
    elif user and not user.is_admin():
        query = query.filter(Storage.project_id.in_(_joined_project_ids(user)))
    if namespace:
        query = query.filter(Storage.namespace.like('%{}%'.format(namespace)))

    for storage in query.order_by(Storage.id.desc()).all():
        if not _storage_is_ready(storage):
            continue
        if storage.project and not user_can_access_project(user, storage.project):
            continue
        namespaces = split_volume_mount(storage.namespace)
        if namespace and namespace not in namespaces:
            continue
        chosen_namespace = namespace if namespace in namespaces else (namespaces[0] if namespaces else namespace)
        pvc_name = storage.pvc_name or storage.name
        mount_path = _storage_mount_path(storage)
        mount_expr = '{}(storage):{}'.format(pvc_name, mount_path)
        _append_unique(items, {
            "id": storage.id,
            "name": storage.name,
            "label": storage.label or storage.name,
            "source": "storage",
            "project": storage.project.name if storage.project else '',
            "volume_type": "storage",
            "storage_type": storage.storage_type,
            "namespace": chosen_namespace,
            "pvc_name": pvc_name,
            "mount_path": mount_path,
            "mount_expr": mount_expr,
            "checked": mount_expr in checked_exprs,
            "status": storage.status,
        }, seen_pvc, seen_mount)

    return items


def available_volume_choices(user, project=None, namespace=None, current_volume_mount='', include_system=True):
    choices = []
    for item in available_volume_items(user, project, namespace, current_volume_mount, include_system):
        source = '系统' if item.get('source') == 'system' else '存储'
        project_name = project.name if project else ''
        if not project_name and item.get('source') == 'storage':
            project_name = item.get('project') or ''
        label = '{} | {} | {}'.format(source, item.get('label') or item.get('name'), item.get('mount_path'))
        if project_name and item.get('source') == 'storage':
            label = '{} | {}'.format(project_name, label)
        choices.append([item['mount_expr'], label])
    return choices


def filter_selected_volume_mount(user, project, selected_volume_mount, namespace=None):
    selected = split_volume_mount(selected_volume_mount)
    if not selected:
        return ''
    if not project or not user_can_access_project(user, project):
        return ''
    allowed_items = available_volume_items(user, project, namespace=namespace, include_system=True)
    allowed = {}
    for item in allowed_items:
        mount_expr = item['mount_expr']
        keys = [
            mount_expr,
            item.get('name'),
            item.get('label'),
            item.get('pvc_name'),
        ]
        for key in keys:
            if key:
                allowed[str(key).strip()] = mount_expr
    return ','.join([allowed[expr] for expr in selected if expr in allowed])
