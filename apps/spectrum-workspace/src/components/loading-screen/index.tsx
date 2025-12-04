import { Loader2 } from 'lucide-react';
import { cn } from '@/lib/utils';

interface LoadingScreenProps {
  message?: string;
  className?: string;
  size?: 'sm' | 'md' | 'lg';
}

export function LoadingScreen({ 
  message = 'Loading...', 
  className,
  size = 'lg' 
}: LoadingScreenProps) {
  const sizeClasses = {
    sm: 'h-6 w-6',
    md: 'h-10 w-10',
    lg: 'h-16 w-16'
  };

  return (
    <div 
      className={cn(
        'flex flex-col items-center justify-center min-h-screen bg-background',
        className
      )}
    >
      {/* Neural Network Animation */}
      <div className="relative">
        {/* Outer ring */}
        <div className="absolute inset-0 rounded-full border-2 border-primary/20 animate-ping" />
        
        {/* Middle ring */}
        <div className="absolute inset-2 rounded-full border border-primary/40 animate-pulse" />
        
        {/* Spinner */}
        <Loader2 
          className={cn(
            sizeClasses[size],
            'text-primary animate-spin'
          )} 
        />
      </div>

      {/* Loading text */}
      <p className="mt-6 text-sm text-muted-foreground animate-pulse">
        {message}
      </p>

      {/* Neurectomy branding */}
      <div className="mt-8 flex items-center gap-2 text-xs text-muted-foreground/60">
        <div className="h-1 w-1 rounded-full bg-primary animate-pulse" />
        <span>NEURECTOMY</span>
        <div className="h-1 w-1 rounded-full bg-primary animate-pulse delay-75" />
      </div>
    </div>
  );
}

export default LoadingScreen;
