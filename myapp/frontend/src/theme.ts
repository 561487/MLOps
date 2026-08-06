interface IThemeConfig {
    [key: string]: string
}

const baseLightTheme = {
    '--app-bg': '#f5f7fb',
    '--app-surface': '#ffffff',
    '--app-surface-elevated': '#ffffff',
    '--app-text': '#1f2329',
    '--app-text-secondary': '#4e5969',
    '--app-border': '#e5e6eb',
    '--app-hover': '#f2f6ff',
};

const star: IThemeConfig = {
    ...baseLightTheme,
    '--ant-primary-color': '#1E1653',
    '--ant-primary-color-hover': '#1E1653',
    '--ant-primary-color-active': '#096dd9',
    '--ant-primary-color-outline': 'rgba(24, 144, 255, 0.2)',
    '--ant-primary-1': '#1E16531a',
    '--ant-primary-2': '#bae7ff',
    '--ant-primary-3': '#91d5ff',
    '--ant-primary-4': '#69c0ff',
    '--ant-primary-5': '#1E1653',
    '--ant-primary-6': '#1E1653',
    '--ant-primary-7': '#096dd9',
    '--ant-primary-color-deprecated-pure': '',
    '--ant-primary-color-deprecated-l-35': '#cbe6ff',
    '--ant-primary-color-deprecated-l-20': '#7ec1ff',
    '--ant-primary-color-deprecated-t-20': '#46a6ff',
    '--ant-primary-color-deprecated-t-50': '#8cc8ff',
    '--ant-primary-color-deprecated-f-12': 'rgba(24, 144, 255, 0.12)',
    '--ant-primary-color-active-deprecated-f-30': 'rgba(230, 247, 255, 0.3)',
    '--ant-primary-color-active-deprecated-d-02': '#dcf4ff',
    '--ant-success-color': '#52c41a',
    '--ant-success-color-hover': '#73d13d',
    '--ant-success-color-active': '#389e0d',
    '--ant-success-color-outline': 'rgba(82, 196, 26, 0.2)',
    '--ant-success-color-deprecated-bg': '#f6ffed',
    '--ant-success-color-deprecated-border': '#b7eb8f',
    '--ant-error-color': '#ff4d4f',
    '--ant-error-color-hover': '#ff7875',
    '--ant-error-color-active': '#d9363e',
    '--ant-error-color-outline': 'rgba(255, 77, 79, 0.2)',
    '--ant-error-color-deprecated-bg': '#fff2f0',
    '--ant-error-color-deprecated-border': '#ffccc7',
    '--ant-warning-color': '#faad14',
    '--ant-warning-color-hover': '#ffc53d',
    '--ant-warning-color-active': '#d48806',
    '--ant-warning-color-outline': 'rgba(250, 173, 20, 0.2)',
    '--ant-warning-color-deprecated-bg': '#fffbe6',
    '--ant-warning-color-deprecated-border': '#ffe58f',
    '--ant-info-color': '#1890ff',
    '--ant-info-color-deprecated-bg': '#e6f7ff',
    '--ant-info-color-deprecated-border': '#91d5ff',
    '--ant-link': '#8e264f',
};

const blue: IThemeConfig = {
    ...baseLightTheme,
    '--ant-primary-color': '#1672fa',
    '--ant-primary-color-hover': '#1672fa',
    '--ant-primary-color-active': '#096dd9',
    '--ant-primary-color-outline': 'rgba(24, 144, 255, 0.2)',
    '--ant-primary-1': '#e6f7ff',
    '--ant-primary-2': '#bae7ff',
    '--ant-primary-3': '#91d5ff',
    '--ant-primary-4': '#69c0ff',
    '--ant-primary-5': '#1672fa',
    '--ant-primary-6': '#1672fa',
    '--ant-primary-7': '#096dd9',
    '--ant-primary-color-deprecated-pure': '',
    '--ant-primary-color-deprecated-l-35': '#cbe6ff',
    '--ant-primary-color-deprecated-l-20': '#7ec1ff',
    '--ant-primary-color-deprecated-t-20': '#46a6ff',
    '--ant-primary-color-deprecated-t-50': '#8cc8ff',
    '--ant-primary-color-deprecated-f-12': 'rgba(24, 144, 255, 0.12)',
    '--ant-primary-color-active-deprecated-f-30': 'rgba(230, 247, 255, 0.3)',
    '--ant-primary-color-active-deprecated-d-02': '#dcf4ff',
    '--ant-success-color': '#52c41a',
    '--ant-success-color-hover': '#73d13d',
    '--ant-success-color-active': '#389e0d',
    '--ant-success-color-outline': 'rgba(82, 196, 26, 0.2)',
    '--ant-success-color-deprecated-bg': '#f6ffed',
    '--ant-success-color-deprecated-border': '#b7eb8f',
    '--ant-error-color': '#ff4d4f',
    '--ant-error-color-hover': '#ff7875',
    '--ant-error-color-active': '#d9363e',
    '--ant-error-color-outline': 'rgba(255, 77, 79, 0.2)',
    '--ant-error-color-deprecated-bg': '#fff2f0',
    '--ant-error-color-deprecated-border': '#ffccc7',
    '--ant-warning-color': '#faad14',
    '--ant-warning-color-hover': '#ffc53d',
    '--ant-warning-color-active': '#d48806',
    '--ant-warning-color-outline': 'rgba(250, 173, 20, 0.2)',
    '--ant-warning-color-deprecated-bg': '#fffbe6',
    '--ant-warning-color-deprecated-border': '#ffe58f',
    '--ant-info-color': '#1890ff',
    '--ant-info-color-deprecated-bg': '#e6f7ff',
    '--ant-info-color-deprecated-border': '#91d5ff',
    '--ant-link': '#1672fa',
};

const dark: IThemeConfig = {
    '--app-bg': '#111827',
    '--app-surface': '#172033',
    '--app-surface-elevated': '#202b40',
    '--app-text': '#f3f7ff',
    '--app-text-secondary': '#a9b7cc',
    '--app-border': '#314057',
    '--app-hover': '#213653',
    '--ant-primary-color': '#5ec7ff',
    '--ant-primary-color-hover': '#7bd4ff',
    '--ant-primary-color-active': '#2fa7e8',
    '--ant-primary-color-outline': 'rgba(94, 199, 255, 0.22)',
    '--ant-primary-1': 'rgba(94, 199, 255, 0.16)',
    '--ant-primary-2': '#174d6b',
    '--ant-primary-3': '#226f95',
    '--ant-primary-4': '#35a5d4',
    '--ant-primary-5': '#5ec7ff',
    '--ant-primary-6': '#5ec7ff',
    '--ant-primary-7': '#8adfff',
    '--ant-link': '#79d2ff',
};

const themesCollection: Record<TThemeType, IThemeConfig> = { star, dark, blue };

export type TThemeType = 'dark' | 'blue' | 'star'

export const THEME_STORAGE_KEY = 'myapp-theme'

export const isThemeType = (theme?: string | null): theme is TThemeType => {
    return theme === 'dark' || theme === 'blue' || theme === 'star'
}

export const getInitialTheme = (defaultTheme: TThemeType): TThemeType => {
    const storedTheme = window.localStorage.getItem(THEME_STORAGE_KEY)

    if (isThemeType(storedTheme)) {
        return storedTheme
    }

    return defaultTheme
}

export const setTheme = (theme: TThemeType, persist = false) => {
    const nextTheme = themesCollection[theme];

    Object.keys(nextTheme).forEach((key) => {
        document.documentElement.style.setProperty(key, nextTheme[key]);
    });

    document.documentElement.setAttribute('data-theme', theme);

    if (persist) {
        window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    }
};
