"""Frontend Design Skill - Vite + Bun + TypeScript + React boilerplate and best practices"""

from ..base import BaseSkill, SkillMetadata, SkillResult


class FrontendDesignSkill(BaseSkill):
    """Skill for setting up and developing modern frontend applications with Vite + Bun + TypeScript + React"""

    def get_metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="frontend-design",
            description="Set up and develop modern frontend apps with Vite + Bun + TypeScript + React",
            version="1.0.0",
            author="SEPilot",
            triggers=[
                "frontend", "react", "vite", "web app", "웹 개발", "프론트엔드",
                "create react", "react app", "typescript react", "react project",
                "web service", "웹 서비스", "SPA", "single page", "UI 개발",
                "bun create", "vite project", "react setup", "web frontend"
            ],
            category="frontend"
        )

    def execute(self, input_text: str, context: dict) -> SkillResult:
        """Execute frontend design skill"""
        frontend_prompt = """## Frontend Development Guidelines (Vite + Bun + TypeScript + React)

### Quick Start - Project Setup

프로젝트를 새로 생성할 때 다음 명령어를 사용하세요:

```bash
# Bun으로 Vite + React + TypeScript 프로젝트 생성
bun create vite <project-name> --template react-ts

# 프로젝트 디렉토리로 이동
cd <project-name>

# 의존성 설치
bun install

# 개발 서버 시작
bun run dev
```

### Recommended Project Structure

```
<project-name>/
├── public/                    # Static assets
│   └── favicon.ico
├── src/
│   ├── assets/               # Images, fonts, etc.
│   ├── components/           # Reusable UI components
│   │   ├── common/           # Shared components (Button, Input, Modal)
│   │   ├── layout/           # Layout components (Header, Footer, Sidebar)
│   │   └── features/         # Feature-specific components
│   ├── hooks/                # Custom React hooks
│   ├── pages/                # Page components (route-level)
│   ├── services/             # API calls and external services
│   ├── stores/               # State management (Zustand/Jotai)
│   ├── styles/               # Global styles and themes
│   │   ├── globals.css
│   │   └── variables.css
│   ├── types/                # TypeScript type definitions
│   ├── utils/                # Utility functions
│   ├── App.tsx               # Root component
│   ├── main.tsx              # Entry point
│   └── vite-env.d.ts         # Vite type declarations
├── index.html
├── package.json
├── tsconfig.json
├── tsconfig.node.json
├── vite.config.ts
├── .eslintrc.cjs
├── .prettierrc
└── README.md
```

### Essential Dependencies

```bash
# Routing
bun add react-router-dom

# State Management (choose one)
bun add zustand                    # Simple, lightweight
# or
bun add jotai                      # Atomic state management

# Styling (choose one or combine)
bun add tailwindcss postcss autoprefixer    # Utility-first CSS
# or
bun add @emotion/react @emotion/styled       # CSS-in-JS

# UI Component Library (optional)
bun add @radix-ui/react-dialog @radix-ui/react-dropdown-menu  # Headless UI
# or
bun add @mantine/core @mantine/hooks          # Full-featured

# Form Handling
bun add react-hook-form zod @hookform/resolvers

# Data Fetching
bun add @tanstack/react-query axios

# Dev Dependencies
bun add -d @types/node prettier eslint-plugin-react-hooks
```

### Tailwind CSS Setup (Recommended)

```bash
bun add -d tailwindcss postcss autoprefixer
bunx tailwindcss init -p
```

**tailwind.config.js:**
```javascript
/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
```

**src/styles/globals.css:**
```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

### TypeScript Best Practices

```typescript
// 1. Component Props 정의
interface ButtonProps {
  variant?: 'primary' | 'secondary' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  disabled?: boolean;
  onClick?: () => void;
  children: React.ReactNode;
}

// 2. Generic 컴포넌트
interface ListProps<T> {
  items: T[];
  renderItem: (item: T) => React.ReactNode;
  keyExtractor: (item: T) => string;
}

function List<T>({ items, renderItem, keyExtractor }: ListProps<T>) {
  return (
    <ul>
      {items.map((item) => (
        <li key={keyExtractor(item)}>{renderItem(item)}</li>
      ))}
    </ul>
  );
}

// 3. API Response 타입
interface ApiResponse<T> {
  data: T;
  status: 'success' | 'error';
  message?: string;
}

// 4. Event Handler 타입
const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
  setValue(e.target.value);
};

const handleSubmit = (e: React.FormEvent<HTMLFormElement>) => {
  e.preventDefault();
  // ...
};
```

### React Component Patterns

```tsx
// 1. Functional Component with TypeScript
import { useState, useCallback, useMemo } from 'react';

interface UserCardProps {
  user: User;
  onSelect?: (user: User) => void;
}

export function UserCard({ user, onSelect }: UserCardProps) {
  const [isHovered, setIsHovered] = useState(false);

  const handleClick = useCallback(() => {
    onSelect?.(user);
  }, [user, onSelect]);

  const displayName = useMemo(() => {
    return `${user.firstName} ${user.lastName}`;
  }, [user.firstName, user.lastName]);

  return (
    <div
      className="p-4 rounded-lg border hover:shadow-lg transition-shadow"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
    >
      <h3 className="font-semibold">{displayName}</h3>
      <p className="text-gray-600">{user.email}</p>
    </div>
  );
}

// 2. Custom Hook
function useLocalStorage<T>(key: string, initialValue: T) {
  const [storedValue, setStoredValue] = useState<T>(() => {
    try {
      const item = window.localStorage.getItem(key);
      return item ? JSON.parse(item) : initialValue;
    } catch {
      return initialValue;
    }
  });

  const setValue = useCallback((value: T | ((val: T) => T)) => {
    try {
      const valueToStore = value instanceof Function ? value(storedValue) : value;
      setStoredValue(valueToStore);
      window.localStorage.setItem(key, JSON.stringify(valueToStore));
    } catch (error) {
      console.error(error);
    }
  }, [key, storedValue]);

  return [storedValue, setValue] as const;
}

// 3. Context Provider Pattern
interface ThemeContextType {
  theme: 'light' | 'dark';
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeContextType | null>(null);

export function useTheme() {
  const context = useContext(ThemeContext);
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider');
  }
  return context;
}
```

### API Integration Pattern

```typescript
// services/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api',
  timeout: 10000,
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

export { api };

// services/userService.ts
import { api } from './api';

export const userService = {
  getAll: () => api.get<User[]>('/users'),
  getById: (id: string) => api.get<User>(`/users/${id}`),
  create: (data: CreateUserDto) => api.post<User>('/users', data),
  update: (id: string, data: UpdateUserDto) => api.put<User>(`/users/${id}`, data),
  delete: (id: string) => api.delete(`/users/${id}`),
};

// hooks/useUsers.ts (with React Query)
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { userService } from '../services/userService';

export function useUsers() {
  return useQuery({
    queryKey: ['users'],
    queryFn: () => userService.getAll().then(res => res.data),
  });
}

export function useCreateUser() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: userService.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['users'] });
    },
  });
}
```

### State Management with Zustand

```typescript
// stores/useAuthStore.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  login: (user: User, token: string) => void;
  logout: () => void;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set) => ({
      user: null,
      token: null,
      isAuthenticated: false,
      login: (user, token) => set({ user, token, isAuthenticated: true }),
      logout: () => set({ user: null, token: null, isAuthenticated: false }),
    }),
    { name: 'auth-storage' }
  )
);
```

### Routing Setup

```tsx
// App.tsx
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const queryClient = new QueryClient();

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<Layout />}>
            <Route index element={<HomePage />} />
            <Route path="dashboard" element={<ProtectedRoute><DashboardPage /></ProtectedRoute>} />
            <Route path="users/:id" element={<UserDetailPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  );
}

// Protected Route Component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuthStore();

  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }

  return <>{children}</>;
}
```

### vite.config.ts Optimization

```typescript
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
      '@components': path.resolve(__dirname, './src/components'),
      '@hooks': path.resolve(__dirname, './src/hooks'),
      '@services': path.resolve(__dirname, './src/services'),
      '@stores': path.resolve(__dirname, './src/stores'),
      '@types': path.resolve(__dirname, './src/types'),
      '@utils': path.resolve(__dirname, './src/utils'),
    },
  },
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom', 'react-router-dom'],
          ui: ['@radix-ui/react-dialog', '@radix-ui/react-dropdown-menu'],
        },
      },
    },
  },
});
```

### tsconfig.json Path Aliases

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "useDefineForClassFields": true,
    "lib": ["ES2020", "DOM", "DOM.Iterable"],
    "module": "ESNext",
    "skipLibCheck": true,
    "moduleResolution": "bundler",
    "allowImportingTsExtensions": true,
    "resolveJsonModule": true,
    "isolatedModules": true,
    "noEmit": true,
    "jsx": "react-jsx",
    "strict": true,
    "noUnusedLocals": true,
    "noUnusedParameters": true,
    "noFallthroughCasesInSwitch": true,
    "baseUrl": ".",
    "paths": {
      "@/*": ["src/*"],
      "@components/*": ["src/components/*"],
      "@hooks/*": ["src/hooks/*"],
      "@services/*": ["src/services/*"],
      "@stores/*": ["src/stores/*"],
      "@types/*": ["src/types/*"],
      "@utils/*": ["src/utils/*"]
    }
  },
  "include": ["src"],
  "references": [{ "path": "./tsconfig.node.json" }]
}
```

### Performance Tips

1. **Lazy Loading**: Use `React.lazy()` and `Suspense` for code splitting
2. **Memoization**: Use `useMemo`, `useCallback`, `React.memo` appropriately
3. **Virtual Lists**: Use `@tanstack/react-virtual` for long lists
4. **Image Optimization**: Use `loading="lazy"` and proper image formats
5. **Bundle Analysis**: Use `rollup-plugin-visualizer` to analyze bundle size

위 가이드라인을 따라 현대적이고 확장 가능한 프론트엔드 애플리케이션을 개발하세요.
"""
        return SkillResult(
            success=True,
            message="Frontend design skill activated - Vite + Bun + TypeScript + React best practices",
            prompt_injection=frontend_prompt
        )
