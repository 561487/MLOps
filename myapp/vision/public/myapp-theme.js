(function () {
  var params = new URLSearchParams(window.location.search);
  var storedTheme = '';

  try {
    storedTheme = localStorage.getItem('myapp-theme') || '';
  } catch (error) {}

  var theme = params.get('theme') || storedTheme || 'star';
  document.documentElement.setAttribute('data-theme', theme === 'dark' ? 'dark' : 'light');

  function injectDarkStyle() {
    if (document.getElementById('myapp-appbuilder-dark-style')) return;

    var style = document.createElement('style');
    style.id = 'myapp-appbuilder-dark-style';
    style.textContent = [
      "html[data-theme='dark']{--app-bg:#111827;--app-surface:#172033;--app-surface-elevated:#202b40;--app-text:#f3f7ff;--app-text-secondary:#a9b7cc;--app-border:#314057;--app-hover:#213653;color-scheme:dark;background:#111827!important;color:#f3f7ff!important}",
      "html[data-theme='dark'] body,html[data-theme='dark'] #root,html[data-theme='dark'] #app{background:#111827!important;color:#f3f7ff!important}",
      "html[data-theme='dark'] body *{border-color:#314057!important}",
      "html[data-theme='dark'] .ant-layout,html[data-theme='dark'] .ant-card,html[data-theme='dark'] .ant-menu,html[data-theme='dark'] .ant-table,html[data-theme='dark'] .ant-table-container,html[data-theme='dark'] .ant-table-thead>tr>th,html[data-theme='dark'] .ant-table-tbody>tr>td,html[data-theme='dark'] .ant-dropdown-menu,html[data-theme='dark'] .ant-popover-inner,html[data-theme='dark'] .ant-select-dropdown,html[data-theme='dark'] .ant-modal-content,html[data-theme='dark'] .ant-drawer-content{background:#172033!important;color:#f3f7ff!important;border-color:#314057!important}",
      "html[data-theme='dark'] .ant-input,html[data-theme='dark'] .ant-input-affix-wrapper,html[data-theme='dark'] .ant-select-selector,html[data-theme='dark'] .ant-btn-default,html[data-theme='dark'] .ant-picker,html[data-theme='dark'] .ant-input-number{background:#202b40!important;color:#f3f7ff!important;border-color:#314057!important}",
      "html[data-theme='dark'] .ant-input::placeholder,html[data-theme='dark'] .ant-empty-description,html[data-theme='dark'] .ant-select-selection-placeholder{color:#a9b7cc!important}",
      "html[data-theme='dark'] .ms-Stack,html[data-theme='dark'] .ms-CommandBar,html[data-theme='dark'] .ms-CommandBar-primaryCommand,html[data-theme='dark'] .ms-Button,html[data-theme='dark'] .ms-TextField-fieldGroup,html[data-theme='dark'] .ms-SearchBox,html[data-theme='dark'] .ms-Dropdown-title,html[data-theme='dark'] .ms-ComboBox,html[data-theme='dark'] .ms-ComboBox-Input{background:#172033!important;color:#f3f7ff!important;border-color:#314057!important}",
      "html[data-theme='dark'] .ms-Button:hover,html[data-theme='dark'] .ms-CommandBarItem-link:hover{background:#213653!important;color:#f3f7ff!important}",
      "html[data-theme='dark'] .react-flow,html[data-theme='dark'] .react-flow__renderer,html[data-theme='dark'] .react-flow__pane,html[data-theme='dark'] .react-flow__viewport,html[data-theme='dark'] .react-flow__zoompane,html[data-theme='dark'] .react-flow__selectionpane{background:#111827!important;color:#f3f7ff!important}",
      "html[data-theme='dark'] .react-flow__node,html[data-theme='dark'] .react-flow__node-default,html[data-theme='dark'] [class*=module],html[data-theme='dark'] [class*=Module],html[data-theme='dark'] [class*=editor],html[data-theme='dark'] [class*=Editor]{background:#172033!important;color:#f3f7ff!important;border-color:#314057!important}",
      "html[data-theme='dark'] .react-flow__edge-path,html[data-theme='dark'] .react-flow__connection-path{stroke:#7f8ca3!important}",
      "html[data-theme='dark'] .react-flow__controls-button{background:#202b40!important;color:#f3f7ff!important;border-color:#314057!important}",
      "html[data-theme='dark'] svg,html[data-theme='dark'] path{color:inherit}"
    ].join('');
    document.head.appendChild(style);
  }

  injectDarkStyle();
  setTimeout(injectDarkStyle, 0);
  setTimeout(injectDarkStyle, 250);
})();
